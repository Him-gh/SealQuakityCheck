
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Ensure project root is on sys.path so relative imports work when launched from other CWDs
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
	sys.path.append(ROOT_DIR)

from ui.main_window import MainWindow


def main():
	# Enable high-DPI scaling where available
	try:
		QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
		QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
	except Exception:
		pass

	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
