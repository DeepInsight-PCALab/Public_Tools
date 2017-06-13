import os

def print_exec_cmd(cmd):
    print '------------------------%s------------------------------' % cmd
    os.system(cmd)

cmd = 'sudo apt-get update'
print_exec_cmd(cmd)

cmd = '''sudo apt-get install wget << EOF
Y
'''
print_exec_cmd(cmd)


cmd = 'wget -qO- https://raw.github.com/ma6174/vim/master/setup.sh | sh -x'
print_exec_cmd(cmd)


cmd = '''sudo apt-get install expect << EOF
Y
'''
print_exec_cmd(cmd)


cmd = '''sudo apt-get install git << EOF
Y
'''
print_exec_cmd(cmd)
