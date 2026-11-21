"""Retain an orange angle; green, blue and black are center-only, with explicit evidence quality."""
import math

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def complete_orientation(cls,result,defaults=None):
    if cls!=2:
        result.update(pose_available=False,pose_valid=False,yaw_world_deg=None,yaw_source='not_requested',yaw_measured=False,pose_quality='center_only')
        return result
    period=180
    fit=result.get('model_fit',{});angle=result.get('yaw_world_deg')
    if result.get('pose_valid') and finite(angle):
        source='accepted_single_solution';quality='fitted';measured=True
    elif fit.get('status') in (1,2) and finite(fit.get('yaw_rad')):
        angle=math.degrees(fit['yaw_rad']);source='accepted_model_best_of_multiple';quality='ambiguous_best';measured=True
    elif finite(fit.get('best_effort',{}).get('yaw_rad')):
        angle=math.degrees(fit['best_effort']['yaw_rad']);source='rejected_model_best_effort';quality='best_effort';measured=False
    else:
        angle=(defaults or {}).get(cls,0.)
        if not finite(angle):raise ValueError('default angles must be finite degrees')
        source='configured_default';quality='default';measured=False
    result.update(pose_available=True,yaw_world_deg=float(angle)%period,yaw_period_deg=period,yaw_source=source,yaw_measured=measured,pose_quality=quality)
    if quality!='fitted':result['pose_valid']=False
    return result


def orange_yaw_cdeg(metric):
    """Axle-frame model angle, 180-degree symmetry, integer centidegrees."""
    angle=metric.get('yaw_world_deg')
    if not finite(angle) or metric.get('yaw_source')=='configured_default':return 0
    return int(round(float(angle)*100.))%18000
