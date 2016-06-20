Actually, I wanna use machine learning algotithms to build some COOL applications. Then I have an idea that signing up the library website of our university without input of validation code because the validation code can be recognized by trained ML model, just like MNIST data. WOW, that's really COOL, act like a HACKER !!!

In the project:
crossBjfuLib.py -- Practicing how to login the website using Python Crawler and find the best split rate for an image.
GetManyValPictures.py -- Just as it's name implied, it will get many pictures for later training model.
gif2gray.py -- Try split all images of the set into four parts(a number per part), and then save greyscale image into local files. (But failed, need to change stratigies)

Login website : http://202.204.121.41:8080/reader/login.php
Validation url : http://202.204.121.41:8080/reader/captcha.php
Target : http://202.204.121.41:8080/reader/redr_verify.php
Final info page : http://202.204.121.41:8080/reader/redr_info.php