# -*- coding: utf-8 -*-
"""Rescue-vision UART packet codec (standalone / 独立实现).

兼容 Python 3.6+，只依赖标准库 ``struct`` 和 ``math``，不引用工程内任何
其他模块，可单独复制、单元测试或在板端直接使用。

Wire format / 帧格式
--------------------
::

    +--------+--------+--------------------------+------------------+
    | AA 55  | N:u8   | N records (7 bytes each) | CRC:u16 (LE)     |
    +--------+--------+--------------------------+------------------+

Record / 记录（``struct`` 格式 ``"<Bhhh"``，小端）::

    type:u8   x_mm:i16   y_mm:i16   yaw:i16

* ``N`` == 0..32  -> 正常帧，后面跟 ``N`` 条记录。
* ``N`` == 33..255 -> 线上非法，:func:`decode` 直接拒绝。

CRC 为 CRC-16/CCITT-FALSE：poly 0x1021，init 0xFFFF，输入/输出均不反射，
xorout 0x0000；只覆盖 ``N`` 与全部记录（即 ``packet[2:-2]``），不覆盖
``AA 55`` 帧头和 CRC 字段本身。

type 4/6 表示己方物资子区、5/7 表示己方伤员子区，各为300x300mm中心。yaw_cdeg 是场内指向区内的车体法线角，
单位0.01度，范围[-18000,18000)。橙色type2使用模型朝向，180度对称周期；类型0/1/3线上yaw仍为0。非法输入抛ValueError。
"""

import math
import struct

__all__ = [
    "HEADER",
    "MAX_TARGETS",
    "TYPE_MIN",
    "TYPE_MAX",
    "RECORD_SIZE",
    "crc16",
    "encode",
    "decode",
]

HEADER = b"\xaa\x55"

MAX_TARGETS = 32          # 正常帧最多记录数
TYPE_MIN = 0              # 合法 type 下界
TYPE_MAX = 7              # 合法 type 上界（闭区间）
RECORD_SIZE = 7           # <Bhhh 的字节数
INT16_MIN = -32768
INT16_MAX = 32767
MIN_PACKET_SIZE = 3 + 2   # 帧头 + N + CRC，零目标帧的最小长度

_RECORD = struct.Struct("<Bhhh")
_CRC = struct.Struct("<H")

assert _RECORD.size == RECORD_SIZE


def _crc_table():
    # CCITT-FALSE is non-reflected; compute the fixed 256-byte table once.
    table = []
    for byte in range(256):
        remainder = byte << 8
        for _ in range(8):
            remainder = ((remainder << 1) ^ (0x1021 if remainder & 0x8000 else 0)) & 0xFFFF
        table.append(remainder)
    return tuple(table)


_CRC_TABLE = _crc_table()


def crc16(data):
    """CRC-16/CCITT-FALSE / 计算校验和。

    polynomial 0x1021, init 0xFFFF, no reflection, xorout 0.
    参数 ``data`` 必须是 bytes/bytearray/memoryview，否则抛 ``ValueError``。
    返回 0..0xFFFF 的整数。

    官方向量: ``crc16(b"123456789") == 0x29B1``
    """
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif not isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "crc16() expects bytes, got %s" % type(data).__name__
        )

    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


def encode(targets):
    """把目标列表编码成完整帧 / encode targets into one frame.

    ``targets`` 为 list（也接受 tuple），元素为 dict，至少包含键
    ``type`` / ``x_mm`` / ``y_mm``；橙色type2及区域类型4..7可另给yaw_cdeg，其余类型角度强制0。

    校验规则：
      * 元素必须是 dict；
      * ``type`` 必须是 int（不接受 bool）且 0..7；
      * ``x_mm`` / ``y_mm`` 必须是有限整数（接受值为整数的 float）且落在
        int16 范围；bool / NaN / inf / 非整数一律非法；
      * 记录数 0..32；超过 32 不截断，直接抛 ``ValueError``，由应用层调用
        按本机诊断处理；线上仅发送普通目标帧。

    返回 bytes，任何非法输入抛 ``ValueError``。

    官方向量:
      ``encode([]) == AA 55 00 F0 E1``
      ``encode([{"type": 0, "x_mm": 450, "y_mm": -100}])``
      ``== AA 55 01 00 C2 01 9C FF 00 00 02 AF``
    """
    if not isinstance(targets, (list, tuple)):
        raise ValueError(
            "encode() expects a list of dicts, got %s" % type(targets).__name__
        )

    count = len(targets)
    if count > MAX_TARGETS:
        # 不截断：交给应用层处理目标容量
        raise ValueError(
            "too many targets: %d (max %d)" % (count, MAX_TARGETS)
        )

    body = bytearray()
    body.append(count)
    for index, target in enumerate(targets):
        body.extend(_pack_target(target, index))

    return HEADER + bytes(body) + _CRC.pack(crc16(bytes(body)))


def decode(packet):
    """解析完整帧 / decode one frame.

    成功返回目标列表（每项 ``{"type": int, "x_mm": int, "y_mm": int}``），
    零目标帧返回空列表。

    会校验：帧头 ``AA 55``、精确长度、N 的合法区间、CRC、type 区间以及
    橙色type2及区域类型4..7的yaw范围、其余类型yaw为0。任何非法输入都抛 ``ValueError``，
    包括：
      * N 在 33..255；
      * 长度与 N 不匹配、CRC 不匹配；
      * 记录 type 不在 0..7；
      * 橙色type2及区域类型4..7角度超范围，或其余类型yaw非0。
    """
    if isinstance(packet, memoryview):
        packet = packet.tobytes()
    elif not isinstance(packet, (bytes, bytearray)):
        raise ValueError(
            "decode() expects bytes, got %s" % type(packet).__name__
        )
    data = bytes(packet)

    if len(data) < MIN_PACKET_SIZE:
        raise ValueError("frame too short: %d bytes" % len(data))
    if data[:2] != HEADER:
        raise ValueError("bad frame header: %s" % data[:2].hex())

    count = data[2]

    if count > MAX_TARGETS:
        raise ValueError("illegal record count: %d" % count)

    # ---- 精确长度 ----
    expected_size = 3 + count * RECORD_SIZE + 2
    if len(data) != expected_size:
        raise ValueError(
            "frame length mismatch: N=%d expects %d bytes, got %d"
            % (count, expected_size, len(data))
        )

    # ---- CRC 覆盖 N + records ----
    crc_offset = 3 + count * RECORD_SIZE
    expected = crc16(data[2:crc_offset])
    received = _CRC.unpack(data[crc_offset:crc_offset + 2])[0]
    if expected != received:
        raise ValueError(
            "frame crc mismatch: expected 0x%04X, got 0x%04X"
            % (expected, received)
        )

    # ---- 逐条记录校验 ----
    targets = []
    for index in range(count):
        offset = 3 + index * RECORD_SIZE
        type_, x_mm, y_mm, yaw = _RECORD.unpack(
            data[offset:offset + RECORD_SIZE]
        )
        if type_ < TYPE_MIN or type_ > TYPE_MAX:
            raise ValueError(
                "record %d: type %d out of range %d..%d"
                % (index, type_, TYPE_MIN, TYPE_MAX)
            )
        if (type_ in (0,1,3) and yaw != 0) or (type_ not in (0,1,3) and not -18000 <= yaw < 18000):
            raise ValueError(
                "record %d: invalid yaw for record type, got %d"
                % (index, yaw)
            )
        target = {"type": type_, "x_mm": x_mm, "y_mm": y_mm}
        if type_ not in (0,1,3):target["yaw_cdeg"] = yaw
        targets.append(target)

    return targets


# ---------------------------------------------------------------------------
# internal helpers / 内部工具
# ---------------------------------------------------------------------------

def _pack_target(target, index):
    """校验单条目标并打包成7字节记录；只有橙色type2及区域类型4..7使用yaw_cdeg。"""
    if not isinstance(target, dict):
        raise ValueError(
            "target %d must be a dict, got %s"
            % (index, type(target).__name__)
        )

    try:
        type_ = target["type"]
        x_mm = target["x_mm"]
        y_mm = target["y_mm"]
    except KeyError as exc:
        raise ValueError("target %d missing key %r" % (index, exc.args[0]))

    type_ = _as_type(type_, index)
    x_mm = _as_int16(x_mm, "x_mm", index)
    y_mm = _as_int16(y_mm, "y_mm", index)

    # 橙色type2及区域类型4..7启用有符号法线角，其余类型保留零角。
    yaw = _as_int16(target.get("yaw_cdeg", 0), "yaw_cdeg", index) if type_ not in (0,1,3) else 0
    if type_ not in (0,1,3) and not -18000 <= yaw < 18000:
        raise ValueError("zone yaw_cdeg must be in [-18000,18000)")
    return _RECORD.pack(type_, x_mm, y_mm, yaw)


def _as_type(value, index):
    """type 必须是 0..7 的 int（不接受 bool）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "target %d: type must be int, got %s"
            % (index, type(value).__name__)
        )
    if value < TYPE_MIN or value > TYPE_MAX:
        raise ValueError(
            "target %d: type %d out of range %d..%d"
            % (index, value, TYPE_MIN, TYPE_MAX)
        )
    return value


def _as_int16(value, name, index):
    """坐标必须是有限整数且落在 int16 范围（bool/NaN/inf/非整数非法）。"""
    if isinstance(value, bool):
        raise ValueError(
            "target %d: %s must be int, got bool" % (index, name)
        )

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "target %d: %s must be finite, got %r" % (index, name, value)
            )
        if not value.is_integer():
            raise ValueError(
                "target %d: %s must be an integer, got %r"
                % (index, name, value)
            )
        value = int(value)
    elif not isinstance(value, int):
        raise ValueError(
            "target %d: %s must be int, got %s"
            % (index, name, type(value).__name__)
        )

    if value < INT16_MIN or value > INT16_MAX:
        raise ValueError(
            "target %d: %s %d out of int16 range %d..%d"
            % (index, name, value, INT16_MIN, INT16_MAX)
        )
    return value
