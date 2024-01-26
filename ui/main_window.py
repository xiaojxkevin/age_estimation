import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QLabel


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(200, 200)
        self.setWindowTitle('PyQt Window')

        self.create_widgets()

        self.setup_layout()

    def create_widgets(self):
        self.input_box = QLineEdit(self)

        self.button = QPushButton('Generate', self)
        self.button.clicked.connect(self.on_button_click)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('face.jpg'))

    def setup_layout(self):
        v_layout = QVBoxLayout(self)

        h_layout = QHBoxLayout(self)

        h_layout.addWidget(self.input_box)
        h_layout.addWidget(self.button)

        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.label)

        self.setLayout(v_layout)

    def on_button_click(self):
        self.label.setPixmap(QPixmap('face2.jpg'))
        print(self.input_box.text())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
