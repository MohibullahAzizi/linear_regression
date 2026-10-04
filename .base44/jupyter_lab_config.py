# JupyterLab config for the Base44 preview.
# No auth / no XSRF so it is usable inside a cross-origin preview iframe.
c = get_config()

c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/work"
c.ServerApp.allow_root = True
c.ServerApp.allow_remote_access = True

# Open server: no token, no password.
c.ServerApp.token = ""
c.ServerApp.password = ""

# Allow the cross-origin preview iframe to embed us and call the API.
c.ServerApp.allow_origin = "*"
c.ServerApp.tornado_settings = {
    "xsrf_cookies": False,
    "headers": {
        "Content-Security-Policy": "frame-ancestors *",
        "X-Frame-Options": "",
    },
}
