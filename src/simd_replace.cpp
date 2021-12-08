#include <stdio.h>

#include <iostream>

using namespace std;

int main(int argc, char** argv) {

	int no_os_flag = 1;

#ifdef linux

	no_os_flag = 0;

	cout << "It is in Linux OS!" << endl;

#endif

#ifdef _UNIX

	no_os_flag = 0;

	cout << "It is in UNIX OS!" << endl;

#endif

#ifdef __WINDOWS_

	no_os_flag = 0;

	cout << "It is in Windows OS!" << endl;

#endif

#ifdef _WIN32

	no_os_flag = 0;

	cout << "It is in WIN32 OS!" << endl;

#endif

	if (1 == no_os_flag) {

		cout << "No OS Defined ,I do not know what the os is!" << endl;
	}

	return 0;
}