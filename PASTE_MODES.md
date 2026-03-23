# Configurable Paste Modes

The voice dictation tool now supports multiple paste key combinations that can be changed at runtime.

## Available Paste Modes

The application supports three different paste key combinations:

1. **Ctrl+V** - Standard paste (default)
   - Works in most applications (text editors, browsers, etc.)
   
2. **Ctrl+Shift+V** - Terminal paste
   - Required by terminal emulators and some console applications
   
3. **Ctrl+Insert** - Alternative paste
   - Traditional alternative paste shortcut, works in some applications

## Runtime Configuration

You can change the paste mode while the application is running using these keyboard shortcuts:

- **F1** - Cycle to the next paste mode
- **F2** - Cycle to the previous paste mode  
- **F3** - Display the current paste mode

## Startup Configuration

The default paste mode is set in `config.py`:

```python
PASTE_KEYS = 'ctrl,v'  # Default: Ctrl+V
```

To change the default, modify this value before starting the application:

```python
PASTE_KEYS = 'ctrl,shift,v'   # Default: Ctrl+Shift+V
PASTE_KEYS = 'ctrl,insert'     # Default: Ctrl+Insert
```

## Usage Example

1. Start the application - it shows the current paste mode
2. When you need to paste into a terminal, press **F1** to switch to Ctrl+Shift+V
3. The application confirms: "🔄 Paste mode changed to: CTRL+SHIFT+V"
4. Make your recording - it will use the new paste shortcut
5. Press **F3** anytime to see the current mode

## Implementation Details

- Modes are stored in `self.available_paste_modes` 
- Current mode index is stored in `self.paste_mode_index`
- The `_paste_text()` method dynamically uses the current mode: `pyautogui.hotkey(*current_mode)`
- Function keys are handled in the `_on_key_event()` method
- Current mode is displayed on startup and when changed