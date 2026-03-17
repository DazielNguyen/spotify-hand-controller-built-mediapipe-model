# macOS Control Module

## Mục đích

Module `mac_control/` chịu trách nhiệm về việc thực thi các điều khiển trên hệ điều hành macOS từ các gesture đã được phân loại. Module này kết nối gesture với các actions thực tế như điều khiển nhạc (Spotify/Apple Music), volume, và các ứng dụng khác.

## Chức năng chính

### 1. Mac Controller (`control.py`)
- **control.py**: Class điều khiển macOS
  - Nhận action từ gesture classifier
  - Thực thi các system-level controls
  - Hỗ trợ:
    - Play/Pause nhạc
    - Next/Previous track
    - Volume up/down
    - Mở/đóng ứng dụng
    - Keyboard shortcuts

## Cấu trúc

```
mac_control/
└── control.py           # macOS control logic
```

## Sử dụng

```python
from mac_control.control import MacController

# Khởi tạo controller
controller = MacController()

# Thực thi action
controller.execute("play_pause")   # Play/Pause
controller.execute("next_track")  # Next song
controller.execute("prev_track")  # Previous song
controller.execute("volume_up")    # Volume up
controller.execute("volume_down")  # Volume down
```

## Action Mapping

| Action | Gesture | Keyboard Shortcut | Mô tả |
|--------|---------|-------------------|-------|
| `play_pause` | open_palm | `playpause` | Phát/Dừng nhạc |
| `next_track` | peace/swipe_right | `nexttrack` | Bài tiếp theo |
| `prev_track` | swipe_left | `prevtrack` | Bài trước |
| `volume_up` | thumbs_up | `volumeup` | Tăng âm lượng |
| `volume_down` | pointing | `volumedown` | Giảm âm lượng |
| `mute` | fist | `mute` | Tắt tiếng |

## Cách hoạt động

```
Gesture Label (from gesture/)
         │
         ▼
┌─────────────────────┐
│  Action Mapping     │ ───> Map gesture to action
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Safety Layer      │ ───> Threshold, debounce, cooldown
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Execute Shortcut   │ ───> pyAppleScript/keyboard
└─────────────────────┘
         │
         ▼
   macOS Action
```

## Safety Layer

Để tránh false triggers, module này cần implement:

1. **Confidence Threshold**: Chỉ thực thi khi confidence > 0.8
2. **Debounce**: Đợi gesture giữ nguyên trong N frames
3. **Cooldown**: Thời gian chờ giữa các commands (tránh spam)
4. **Gesture Stability**: Yêu cầu gesture ổn định trong ít nhất 3 frames

## Dependencies

- `pyobjc`: Điều khiển macOS system events
- `keyboard`: Gửi keyboard shortcuts
- `appscript` hoặc `osascript`: Điều khiển ứng dụng

## Ghi chú

- Module đang trong giai đoạn placeholder
- Yêu cầu quyền Accessibility trong macOS System Preferences
- Cần test với Spotify và Apple Music
- Nên implement logging để debug
