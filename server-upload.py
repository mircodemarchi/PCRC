"""Server Upload

This is a python program used only as a script.

The purpose of this script is to upload the PCRC files on the PARCO Lab 
servers in a easy way. 
To use this script you have to create a configuration file. You can find a 
template of the configuration file in "<project-root>/spec/defconfig.ini". I 
suggest to copy this file in the <project-root> directory and edit with your 
data. You have to fill the configuration file with:
- Server address;
- Username registered in the server;
- Paths of the folder and file you want to upload relative to the 
<project-root> folder.
- Port of the server ssh connection (optional, default 4080);
- Remote server destination folder (optional, default ~/pcrc);

!!! the paths has to be specified with the ", " separator !!!

Template of the configuration file:
[SERVER-UPLOAD]
Paths  = 
Server = 
User   = 
Port   = 
RemoteFolder = 

Script Usage:
python3 server-upload.py [-h] [--config CONFIG_FILE]

where the CONFIG_FILE is the configuration file path that you have created 
with all your server info and by default is the "<project-root>/config.ini" 
file.

Functions
---------
get_file_instance(file)
    get class File instance of file
compare_hashcode(file1, file2)
    calculate the hashcode of file1 and file2 and return True if equals
get_files_path(dir)
    get the list of files path in the directory tree 
detect_clone(dir)
    detect the clone pairs filepath in the dir directory tree
"""

from paramiko import SSHClient
from scp import SCPClient
import sys
import argparse
import getpass 
import configparser
import os


# Build the argparse.
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    "-c",
                    dest="config_file",
                    required=False,
                    help="Configuration file with all server specification")
args = parser.parse_args()


def get_password(server, username):
    """Get password from console.

    Parameters
    ----------
    server: str
        Server address.
    username: str
        Username registered in the server.
    """
    return getpass.getpass(prompt="{}@{} password: ".format(username, server)) 


def connect(server, user, password, port=4080):
    """Connect SSH and SCP client to the server.

    Parameters
    ----------
    server: str
        Server address.
    user: str
        Username registered in the server.
    password: str
        Password for the server authentication.
    port: int, optional
        Server SSH connection port.

    Returns
    -------
    SCPClient
        SCP object related to the server connection. 
    """

    # Init SSH.
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, port, user, password)

    # Callback that prints the current percentage completed for the connection.
    # def progress(filename, size, sent):
    #     sys.stdout.write("%s\'s progress: %.2f%% \r" 
    #     % (filename, float(sent)/float(size)*100) )
    def progress(filename, size, sent, peername):
        sys.stdout.write("(%s:%s) %s\'s progress: %.2f%%              \r" % 
            (peername[0], peername[1], filename, float(sent)/float(size)*100))

    # SCPCLient takes a paramiko transport.
    scp = SCPClient(ssh.get_transport(), progress4=progress)

    return scp


def close(scp):
    """Free all the script resource allocated.

    Parameters
    ----------
    scp: SCPClient
        Object related to the SCP server connection.
    """

    scp.close()


def upload(scp, folders, destination="~/pcrc"):
    """Upload all the folders to the server.

    Parameters
    ----------
    scp: SCPClient
        Object related to the SCP server connection.
    folders: list
        List of path string of the folders and files to upload.
    destination: str, optional
        Remote destination server path.
    """

    # Upload each folder.
    for folder in folders:
        if not os.path.exists(folder):
            continue

        scp.put(os.path.join(os.path.dirname(__file__), folder), 
            recursive=True, remote_path=destination)

    sys.stdout.write("\033[K") # Clear line.
    print("Completed")


def get_folders_to_upload(paths):
    """Parse folders to upload to the server.

    Parameters
    ----------
    paths: str
        String of the paths of the folders and files to upload.

    Returns
    -------
    list
        List of the paths string of the folders and files to upload.
    """
    return paths.split(", ")


def main(): 
    """Script wrapper"""

    # Parse the arguments.
    config_file = args.config_file

    # Check configuration file.
    if not config_file:
        print("Config file not selected, take the default config file")
        config_file = os.path.join(os.path.dirname(__file__), "config.ini")

    # Check if source directory exist
    if not os.path.exists(config_file):
        print("Path of configuration file specified doesn't exist")
        exit()

    # Init config parser and retrieve data from configuration file.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Set the global data environment.
    paths  = config["SERVER-UPLOAD"]["Paths"]
    server = config["SERVER-UPLOAD"]["Server"]
    user   = config["SERVER-UPLOAD"]["User"]
    port   = config["SERVER-UPLOAD"]["Port"]
    remote_folder = config["SERVER-UPLOAD"]["RemoteFolder"]

    # Check global data environment.
    if not paths:
        print("No path specified to upload in configuration file.")
        exit()

    if not server:
        print("No server address specified in configuration file.")
        exit()

    if not user:
        print("No username specified in configuration file.")
        exit()

    # Get password from user.
    password = get_password(server, user)

    # Connect using SSH.
    if port:
        scp = connect(server, user, password, port)
    else:
        scp = connect(server, user, password)

    # Upload files.
    if remote_folder:
        upload(scp, get_folders_to_upload(paths), remote_folder)
    else:
        upload(scp, get_folders_to_upload(paths))

    # Free memory.
    close(scp)

if __name__ == "__main__":
    main()
