# ED3S'17

Here you can find lecture slides as well as source code and solutions to hands on.

### Jupyter Hub image used for hands - on part

The Docker image with all the dependencies is locted at:

[https://hub.docker.com/r/issuds/jphub/](https://hub.docker.com/r/issuds/jphub/)

To run the image, install Docker and run command:

`[sudo] docker run -p 8000:8000 issuds/jphub python3 -m jupyterhub`

and go to [localhost:8000](localhost:8000). You should see the login page of Jupyter Hub.
Use login `user` and password `pass0123`.

By default, all changes are stored inside the Docker container. In order to access the 
result of your work from outside the container, mount some host folder to the folder in
Docker container. For example, assuming that you have a folder `/home/ubuntu/persistent`
on your host machine, you can run container with such folder mounted to the user folder
in container using:

`[sudo] docker run -v /home/ubuntu/persistent:/home/user/persistent -p 8000:8000 issuds/jphub python3 -m jupyterhub`

Any changes that you make to `persistent` folder will be saved outside the container.
