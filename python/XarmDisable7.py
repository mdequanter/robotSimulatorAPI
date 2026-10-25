from xarm.wrapper import XArmAPI

arm = XArmAPI("10.2.172.20")

result = arm.motion_enable(
    enable=False,
    servo_id=7
)

print("Resultaat:", result)
print("Controllerfout:", arm.get_err_warn_code())

arm.disconnect()