from colorama import Fore, Style, init
def print_header():
    header = r"""
 ______ ___   _____ _   _   ___  _____ 
|  ___/ _ \ /  ___| | | | / _ \|_   _|
| |_ / /_\ \\ `--.| |_| |/ /_\ \ | |  
|  _||  _  | `--. \  _  ||  _  | | |  
| |  | | | |/\__/ / | | || | | |_| |_ 
\_|  \_| |_/\____/\_| |_/\_| |_/\___/ 
 by: @augustxj
    """
    print(Fore.CYAN + header + Style.RESET_ALL)
    print(Fore.YELLOW + "Fashion Image Recommendation System 🚀\n" + Style.RESET_ALL)