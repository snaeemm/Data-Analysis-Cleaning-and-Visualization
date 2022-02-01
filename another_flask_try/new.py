from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return ("This is SPARTA!!")


@app.route("/admin")
def hello_admin():
    return ("Admin is here")


@app.route("/admin2")
def hello_admin2():
    return ("Admin2 is here")

app.run()

