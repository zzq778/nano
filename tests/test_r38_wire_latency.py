"""Wire parity and bounded latency accounting introduced in r38."""
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from rescuevision import uart
from app import latency_quantiles


def reference_crc16(data):
    """Independent pre-r38 bitwise CCITT-FALSE algorithm (not used in production)."""
    remainder = 0xFFFF
    for byte in data:
        remainder ^= byte << 8
        for _ in range(8):
            remainder = ((remainder << 1) ^ (0x1021 if remainder & 0x8000 else 0)) & 0xFFFF
    return remainder


class FastWireContracts(unittest.TestCase):
    def test_known_crc_golden_vectors(self):
        self.assertEqual(uart.crc16(b'123456789'), 0x29B1)
        self.assertEqual(uart.encode([]).hex(), 'aa5500f0e1')
        self.assertEqual(uart.encode([dict(type=0,x_mm=450,y_mm=-100)]).hex(), 'aa550100c2019cff000002af')

    def test_crc_matches_bitwise_reference_for_bursty_and_random_frames(self):
        rng = random.Random(20261121)
        for length in (0, 1, 2, 3, 15, 32, 100, 225, 256):
            for _ in range(30):
                payload = bytes(rng.getrandbits(8) for _ in range(length))
                self.assertEqual(uart.crc16(payload), reference_crc16(payload))
        for count in (0,1,15,32):
            targets = [dict(type=i%8,x_mm=i*37-200,y_mm=i*11-150,
                            yaw_cdeg=(i*29 if i%8 in (2,4,5,6,7) else 0)) for i in range(count)]
            frame = uart.encode(targets)
            decoded=uart.decode(frame)
            self.assertEqual(len(decoded),count)
            for original,received in zip(targets,decoded):
                self.assertEqual(received['type'],original['type'])
                self.assertEqual(received['x_mm'],original['x_mm'])
                self.assertEqual(received['y_mm'],original['y_mm'])
                self.assertEqual(received.get('yaw_cdeg',0),original['yaw_cdeg'])
            self.assertEqual(int.from_bytes(frame[-2:],'little'), reference_crc16(frame[2:-2]))
            with self.assertRaises(ValueError):uart.decode(frame[:-1]+bytes([frame[-1]^1]))
        with self.assertRaisesRegex(ValueError,'too many targets'):
            uart.encode([dict(type=0,x_mm=0,y_mm=0)]*33)

    def test_target_validation_preserves_encoder_error_contract(self):
        invalid = [dict(type=8,x_mm=0,y_mm=0),dict(type=0,x_mm=float('nan'),y_mm=0),
                   dict(type=2,x_mm=0,y_mm=0,yaw_cdeg=18000),dict(type=0,x_mm=2**16,y_mm=0)]
        for target in invalid:
            with self.subTest(target=target):
                try:uart.encode([target])
                except ValueError as expected:message=str(expected)
                else:self.fail('target should be rejected')
                with self.assertRaisesRegex(ValueError, '.*') as caught:
                    uart._pack_target(target,0)
                self.assertEqual(str(caught.exception), message)


class LatencyWindowContracts(unittest.TestCase):
    def test_latency_quantiles_and_empty_window(self):
        self.assertIsNone(latency_quantiles([]))
        self.assertEqual(latency_quantiles([5]), dict(p50_ms=5.,p95_ms=5.,samples=1))
        result=latency_quantiles(range(1,101))
        self.assertEqual((result['p50_ms'],result['p95_ms'],result['samples']),(50.5,95,100))


if __name__ == '__main__':unittest.main()
