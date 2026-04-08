from serial import Serial
from stepper.device import Device
from stepper.stepper_core.parameters import DeviceParams
from stepper.stepper_core.configs import Address

# Connect to motor
serial = Serial("/dev/ttyACM0", 115200, timeout=0.1)
device = Device(
    device_params=DeviceParams(
        serial_connection=serial,
        address=Address(0x01)
    )
)

# Basic movement controls
device.enable()
device.move_cw(1000)  # Move 1000 pulses clockwise
device.move_to(0)     # Move to absolute position 0
device.jog(500)       # Continuous movement at 500 pulses/sec
device.stop()         # Stop movement

# Get various status information
print(f"Position: {device.real_time_position.position}")
print(f"Speed: {device.real_time_speed.speed}")
print(f"Status: Enabled={device.is_enabled}, In Position={device.is_in_position}")
print(f"Voltage: {device.bus_voltage.voltage}")
print(f"Current: {device.phase_current.current}")