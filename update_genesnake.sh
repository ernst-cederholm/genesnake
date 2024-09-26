
bash build_cleanup.sh
python3 -m build
pip uninstall genesnake
pip install dist/genesnake-0.0.1-py3-none-any.whl

# # You can also make and editable install for development work with
# pip install --editable .
# which makes changes to source code take effect immediately,
# without rebuilding and reinstlling.
# For more details, see
# https://setuptools.pypa.io/en/latest/userguide/development_mode.html
