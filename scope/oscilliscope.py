# Enhanced XLR Oscilloscope: Real-time Visualizer for Predictive Coding Kernel
# With FFT analysis, trigger modes, and parameter controls

import torch
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from scipy.fft import fft, fftfreq
import sys

class XLROscilloscope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XLR Predictive Oscilloscope")
        self.resize(1200, 800)
        
        # --- CONFIGURATION ---
        self.PATCH = 32
        self.K = 8
        self.D = 4
        self.I, self.J = 6, 6
        self.α = 0.9
        self.noise_level = 0.05
        self.perturbation_level = 0.1
        
        # --- INIT THOUGHT + WEIGHTS ---
        self.top_vector = torch.randn(self.I * self.J)
        self.top_tensor = self.top_vector.view(self.I, self.J)
        self.W_top = torch.randn(self.I, self.J, self.K, self.D)
        self.W_mid = torch.randn(self.I, self.J, self.K, self.D)
        self.memory2 = torch.zeros(self.K, self.PATCH, self.PATCH, self.D)
        
        # --- Trigger settings ---
        self.trigger_enabled = False
        self.trigger_channel = 0  # 0=phase, 1=error, 2=symbolic
        self.trigger_level = 0.0
        self.trigger_mode = 0  # 0=rising, 1=falling, 2=change
        self.triggered = False
        self.trigger_cooldown = 0
        
        # --- Buffers for streaming data ---
        self.buffer_size = 1000
        self.data_phase = np.zeros(self.buffer_size)
        self.data_error = np.zeros(self.buffer_size)
        self.data_symbol = np.zeros(self.buffer_size)
        self.data_spectral = np.zeros((3, self.buffer_size // 2))
        
        # --- Set up the UI ---
        self.setup_ui()
        
        # --- Timer for updates ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)  # ~30 FPS
        
        # --- Performance tracking ---
        self.last_time = QtCore.QTime.currentTime()
        self.fps = 0
        self.frame_count = 0
    
    def setup_ui(self):
        # Main widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Top panel for plots
        plot_layout = QtWidgets.QGridLayout()
        main_layout.addLayout(plot_layout, stretch=4)
        
        # Create plots
        self.create_plots(plot_layout)
        
        # Bottom panel for controls
        control_panel = QtWidgets.QGroupBox("Control Panel")
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)
        
        # Create control panels
        self.create_parameter_controls(control_layout)
        self.create_trigger_controls(control_layout)
        self.create_info_panel(control_layout)
        
        # Status bar
        self.statusBar().showMessage("XLR Oscilloscope Ready")
    
    def create_plots(self, layout):
        # Time domain plots on the left
        self.plot_phase = pg.PlotWidget(title="Layer 1 Phase Trace (mean θ)")
        self.plot_error = pg.PlotWidget(title="Error Field Intensity (norm)")
        self.plot_symbolic = pg.PlotWidget(title="Symbolic Vector Magnitude")
        
        # Add trigger lines
        self.trigger_line = pg.InfiniteLine(pos=0, angle=0, movable=True, pen='r')
        self.trigger_line.sigPositionChanged.connect(self.update_trigger_level)
        
        # FFT plots on the right
        self.plot_fft = pg.PlotWidget(title="Frequency Spectrum")
        
        # Add plots to layout
        layout.addWidget(self.plot_phase, 0, 0)
        layout.addWidget(self.plot_error, 1, 0)
        layout.addWidget(self.plot_symbolic, 2, 0)
        layout.addWidget(self.plot_fft, 0, 1, 3, 1)  # Spans 3 rows
        
        # Setup curves
        self.curve_phase = self.plot_phase.plot(pen='c', name="Phase")
        self.curve_error = self.plot_error.plot(pen='m', name="Error")
        self.curve_symbol = self.plot_symbolic.plot(pen='y', name="Symbol")
        
        # Setup FFT curves with different colors
        self.fft_curves = [
            self.plot_fft.plot(pen='c', name="Phase FFT"),
            self.plot_fft.plot(pen='m', name="Error FFT"),
            self.plot_fft.plot(pen='y', name="Symbol FFT")
        ]
        
        # Add trigger line to default plot
        self.plot_phase.addItem(self.trigger_line)
        
        # Add legends
        self.plot_fft.addLegend()
        
        # Configure plots
        for plot in [self.plot_phase, self.plot_error, self.plot_symbolic, self.plot_fft]:
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setMouseEnabled(x=True, y=True)
            plot.setMenuEnabled(True)
            
        # Add X and Y labels
        self.plot_phase.setLabel('left', 'Phase', units='rad')
        self.plot_phase.setLabel('bottom', 'Time', units='frames')
        
        self.plot_error.setLabel('left', 'Error', units='norm')
        self.plot_error.setLabel('bottom', 'Time', units='frames')
        
        self.plot_symbolic.setLabel('left', 'Magnitude', units='norm')
        self.plot_symbolic.setLabel('bottom', 'Time', units='frames')
        
        self.plot_fft.setLabel('left', 'Power', units='dB')
        self.plot_fft.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_fft.setLogMode(x=True, y=True)  # Log scale for FFT
    
    def create_parameter_controls(self, layout):
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        layout.addWidget(param_group, stretch=2)
        
        # Memory decay (α)
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.α * 100))
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        self.alpha_label = QtWidgets.QLabel(f"α: {self.α:.2f}")
        param_layout.addRow(self.alpha_label, self.alpha_slider)
        
        # Noise level
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(int(self.noise_level * 100))
        self.noise_slider.valueChanged.connect(self.update_noise)
        self.noise_label = QtWidgets.QLabel(f"Noise: {self.noise_level:.2f}")
        param_layout.addRow(self.noise_label, self.noise_slider)
        
        # Perturbation level
        self.perturb_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.perturb_slider.setRange(0, 100)
        self.perturb_slider.setValue(int(self.perturbation_level * 100))
        self.perturb_slider.valueChanged.connect(self.update_perturbation)
        self.perturb_label = QtWidgets.QLabel(f"Perturbation: {self.perturbation_level:.2f}")
        param_layout.addRow(self.perturb_label, self.perturb_slider)
        
        # Reset button
        self.reset_button = QtWidgets.QPushButton("Reset System")
        self.reset_button.clicked.connect(self.reset_system)
        param_layout.addRow(self.reset_button)
    
    def create_trigger_controls(self, layout):
        trigger_group = QtWidgets.QGroupBox("Trigger Settings")
        trigger_layout = QtWidgets.QFormLayout(trigger_group)
        layout.addWidget(trigger_group, stretch=2)
        
        # Enable trigger
        self.trigger_checkbox = QtWidgets.QCheckBox("Enable Trigger")
        self.trigger_checkbox.setChecked(self.trigger_enabled)
        self.trigger_checkbox.stateChanged.connect(self.toggle_trigger)
        trigger_layout.addRow(self.trigger_checkbox)
        
        # Trigger source
        self.trigger_source = QtWidgets.QComboBox()
        self.trigger_source.addItems(["Phase", "Error", "Symbol"])
        self.trigger_source.currentIndexChanged.connect(self.update_trigger_source)
        trigger_layout.addRow("Source:", self.trigger_source)
        
        # Trigger mode
        self.trigger_mode_combo = QtWidgets.QComboBox()
        self.trigger_mode_combo.addItems(["Rising Edge", "Falling Edge", "Level Change"])
        self.trigger_mode_combo.currentIndexChanged.connect(self.update_trigger_mode)
        trigger_layout.addRow("Mode:", self.trigger_mode_combo)
        
        # Trigger level
        self.trigger_level_spin = QtWidgets.QDoubleSpinBox()
        self.trigger_level_spin.setRange(-10, 10)
        self.trigger_level_spin.setValue(self.trigger_level)
        self.trigger_level_spin.valueChanged.connect(self.update_trigger_level_from_spin)
        trigger_layout.addRow("Level:", self.trigger_level_spin)
        
        # Trigger arm button
        self.arm_button = QtWidgets.QPushButton("Arm Trigger")
        self.arm_button.clicked.connect(self.arm_trigger)
        trigger_layout.addRow(self.arm_button)
        
        # Trigger status
        self.trigger_status = QtWidgets.QLabel("Status: Disarmed")
        trigger_layout.addRow(self.trigger_status)
    
    def create_info_panel(self, layout):
        info_group = QtWidgets.QGroupBox("System Info")
        info_layout = QtWidgets.QFormLayout(info_group)
        layout.addWidget(info_group, stretch=1)
        
        # Current FPS
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        info_layout.addRow(self.fps_label)
        
        # Peak detection
        self.phase_peak_label = QtWidgets.QLabel("Phase Peak: 0 Hz")
        self.error_peak_label = QtWidgets.QLabel("Error Peak: 0 Hz")
        self.symbol_peak_label = QtWidgets.QLabel("Symbol Peak: 0 Hz")
        
        info_layout.addRow(self.phase_peak_label)
        info_layout.addRow(self.error_peak_label)
        info_layout.addRow(self.symbol_peak_label)
        
        # Save Button
        self.save_button = QtWidgets.QPushButton("Save State")
        self.save_button.clicked.connect(self.save_state)
        info_layout.addRow(self.save_button)
    
    def update_alpha(self):
        self.α = self.alpha_slider.value() / 100.0
        self.alpha_label.setText(f"α: {self.α:.2f}")
    
    def update_noise(self):
        self.noise_level = self.noise_slider.value() / 100.0
        self.noise_label.setText(f"Noise: {self.noise_level:.2f}")
    
    def update_perturbation(self):
        self.perturbation_level = self.perturb_slider.value() / 100.0
        self.perturb_label.setText(f"Perturbation: {self.perturbation_level:.2f}")
    
    def toggle_trigger(self, state):
        self.trigger_enabled = (state == QtCore.Qt.Checked)
        if self.trigger_enabled:
            self.trigger_status.setText("Status: Armed")
        else:
            self.trigger_status.setText("Status: Disabled")
            self.triggered = False
    
    def update_trigger_source(self, index):
        self.trigger_channel = index
        # Move trigger line to appropriate plot
        if hasattr(self, 'trigger_line'):
            # Remove from all plots
            for plot in [self.plot_phase, self.plot_error, self.plot_symbolic]:
                if self.trigger_line in plot.items:
                    plot.removeItem(self.trigger_line)
            
            # Add to selected plot
            if index == 0:
                self.plot_phase.addItem(self.trigger_line)
            elif index == 1:
                self.plot_error.addItem(self.trigger_line)
            elif index == 2:
                self.plot_symbolic.addItem(self.trigger_line)
    
    def update_trigger_mode(self, index):
        self.trigger_mode = index
    
    def update_trigger_level_from_spin(self, value):
        self.trigger_level = value
        # Update the trigger line position
        if hasattr(self, 'trigger_line'):
            self.trigger_line.setValue(value)
    
    def update_trigger_level(self):
        if hasattr(self, 'trigger_line'):
            self.trigger_level = self.trigger_line.value()
            # Update spinbox
            self.trigger_level_spin.setValue(self.trigger_level)
    
    def arm_trigger(self):
        self.trigger_enabled = True
        self.trigger_checkbox.setChecked(True)
        self.triggered = False
        self.trigger_status.setText("Status: Armed")
    
    def reset_system(self):
        # Reset the tensor states
        self.top_vector = torch.randn(self.I * self.J)
        self.top_tensor = self.top_vector.view(self.I, self.J)
        self.memory2 = torch.zeros(self.K, self.PATCH, self.PATCH, self.D)
        
        # Reset buffers
        self.data_phase = np.zeros(self.buffer_size)
        self.data_error = np.zeros(self.buffer_size)
        self.data_symbol = np.zeros(self.buffer_size)
        self.data_spectral = np.zeros((3, self.buffer_size // 2))
        
        # Reset trigger
        self.triggered = False
        self.trigger_status.setText("Status: Armed" if self.trigger_enabled else "Status: Disabled")
    
    def save_state(self):
        # Save current state to file
        state = {
            'top_vector': self.top_vector.numpy(),
            'W_top': self.W_top.numpy(),
            'W_mid': self.W_mid.numpy(),
            'memory2': self.memory2.numpy(),
            'data_phase': self.data_phase,
            'data_error': self.data_error,
            'data_symbol': self.data_symbol,
            'params': {
                'alpha': self.α,
                'noise': self.noise_level,
                'perturbation': self.perturbation_level
            }
        }
        
        # Save with numpy
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save State', 
                                                      filter='NumPy Files (*.npz)')[0]
        if filename:
            np.savez(filename, **state)
            self.statusBar().showMessage(f"State saved to {filename}")
    
    def check_trigger(self, current_value, prev_value):
        if not self.trigger_enabled or self.triggered:
            return False
        
        # Apply cooldown to avoid re-triggering too quickly
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            return False
        
        # Check trigger conditions
        if self.trigger_mode == 0:  # Rising edge
            if prev_value < self.trigger_level and current_value >= self.trigger_level:
                self.trigger_cooldown = 30  # ~1 second cooldown at 30fps
                return True
        elif self.trigger_mode == 1:  # Falling edge
            if prev_value > self.trigger_level and current_value <= self.trigger_level:
                self.trigger_cooldown = 30
                return True
        elif self.trigger_mode == 2:  # Level change
            if abs(current_value - prev_value) > self.trigger_level:
                self.trigger_cooldown = 30
                return True
        
        return False
    
    def calculate_fft(self):
        # Sample rate (assuming 30 FPS)
        sample_rate = 30.0
        
        # Calculate FFT for each signal
        for i, data in enumerate([self.data_phase, self.data_error, self.data_symbol]):
            # Use real data for FFT
            yf = fft(data)
            xf = fftfreq(len(data), 1.0/sample_rate)
            
            # Only take positive frequencies and exclude DC (0 Hz)
            positive_mask = xf > 0
            xf = xf[positive_mask]
            yf = yf[positive_mask]
            
            # Calculate magnitude in dB (20*log10)
            # Add small constant to avoid log(0)
            magnitude = 20 * np.log10(np.abs(yf) + 1e-10)
            
            # Store in spectral data
            n_points = min(len(magnitude), self.buffer_size // 2)
            self.data_spectral[i, :n_points] = magnitude[:n_points]
            
            # Find peak frequency (ignore very low frequencies < 0.1 Hz)
            peak_mask = xf > 0.1
            if np.any(peak_mask):
                peak_idx = np.argmax(magnitude[peak_mask])
                peak_freq = xf[peak_mask][peak_idx]
                
                # Update peak frequency labels
                if i == 0:
                    self.phase_peak_label.setText(f"Phase Peak: {peak_freq:.2f} Hz")
                elif i == 1:
                    self.error_peak_label.setText(f"Error Peak: {peak_freq:.2f} Hz")
                elif i == 2:
                    self.symbol_peak_label.setText(f"Symbol Peak: {peak_freq:.2f} Hz")
    
    def update(self):
        # Calculate FPS
        current_time = QtCore.QTime.currentTime()
        elapsed = self.last_time.msecsTo(current_time)
        self.last_time = current_time
        
        if elapsed > 0:
            instant_fps = 1000.0 / elapsed
            self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
            self.frame_count += 1
            
            # Update FPS label every 30 frames
            if self.frame_count % 30 == 0:
                self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # --- XLR Computational Core ---
        perturbation = torch.randn(self.I, self.J) * self.perturbation_level
        self.top_tensor = torch.tanh(self.top_vector.view(self.I, self.J) + perturbation)

        # L3: Top → Mid prediction
        layer3 = torch.einsum('ij,ijkd -> kwhd', self.top_tensor, self.W_top)
        shadow3 = layer3.clone()
        shadow3[..., 0] = (shadow3[..., 0] + torch.pi) % (2 * torch.pi)

        # L2: With memory
        self.memory2 = self.α * self.memory2 + (1 - self.α) * layer3
        layer2 = torch.einsum('ij,ijkd -> kwhd', self.top_tensor, self.W_mid)
        shadow2 = layer2.clone()
        shadow2[..., 0] = (shadow2[..., 0] + torch.pi) % (2 * torch.pi)

        sensor_field = layer2 + torch.randn_like(layer2) * self.noise_level

        error_field = sensor_field + shadow2
        phase_trace = sensor_field[..., 0].mean().item()        # Mean phase θ
        error_mag = torch.norm(error_field).item()              # Global error intensity

        # Symbolic collapse
        collapsed = torch.norm(layer3.mean(dim=(1, 2)), dim=-1).sum().item()

        # --- Check for trigger events ---
        # Get channel to check based on trigger_channel
        channels = [phase_trace, error_mag, collapsed]
        current_value = channels[self.trigger_channel]
        prev_value = self.data_phase[-1] if self.trigger_channel == 0 else \
                     self.data_error[-1] if self.trigger_channel == 1 else \
                     self.data_symbol[-1]
        
        if self.check_trigger(current_value, prev_value):
            self.triggered = True
            self.trigger_status.setText("Status: Triggered!")
            # When triggered, don't update the buffers - freeze the display
            return

        # Update streaming buffers (only if not triggered)
        if not self.triggered:
            self.data_phase = np.roll(self.data_phase, -1)
            self.data_error = np.roll(self.data_error, -1)
            self.data_symbol = np.roll(self.data_symbol, -1)
            self.data_phase[-1] = phase_trace
            self.data_error[-1] = error_mag
            self.data_symbol[-1] = collapsed
            
            # Calculate FFT approximately 5 times per second
            if self.frame_count % 6 == 0:
                self.calculate_fft()

        # Update plots
        # Time domain
        self.curve_phase.setData(self.data_phase)
        self.curve_error.setData(self.data_error)
        self.curve_symbol.setData(self.data_symbol)
        
        # FFT domain - use log frequency scale
        frequencies = fftfreq(self.buffer_size, 1.0/30.0)
        positive_mask = frequencies > 0
        freqs = frequencies[positive_mask]
        
        for i, curve in enumerate(self.fft_curves):
            curve.setData(freqs, self.data_spectral[i, :len(freqs)])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = XLROscilloscope()
    window.show()
    sys.exit(app.exec_())