//  Particle Viewer
//  Copyright (C) 2021 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <chrono>
#include <thread>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <FreeImage.h>

// enable optimus, which choose dedicated GPU if present
extern "C"
{
    _declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    _declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

#define TEST(x)
#include "Font.hpp"

using namespace std;

Font* font;
GLuint progID;
GLuint va, vb;

GLuint LoadShader(const char* VertexSPath, const char* FragmentSPath);

int Init(const int l, const int h)
{
	glewExperimental = GL_TRUE;
	GLenum st = glewInit();

	if (st != GLEW_OK)
	{
		cerr << "GLEW Error: " << glewGetErrorString(st) << endl;
		return -1;
	}

	glViewport(0, 0, l, h);

	clog << "GLEW version: " << glewGetString(GLEW_VERSION) << endl;

	glGenVertexArrays(1, &va);

	glBindVertexArray(va);

	progID = LoadShader("vertex.vsh", "fragment.fsh");
	GLuint progFont = LoadShader("vertexfont.vsh", "fragmentfont.fsh");

	if (!progID || !progFont)
		return -1;

	font = new Font("LiberationSans-Regular.ttf", 24, l, h, progFont);

	if (!font->good())
		return -1;

	double asp = (double)l / (double)h;

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	return 0;
}

void Terminate()
{
	if (font)
		delete font;

	if (vb)
		glDeleteBuffers(1, &vb);
	if (progID)
		glDeleteProgram(progID);
	if (va)
		glDeleteVertexArrays(1, &va);

	glfwTerminate();
}

static void glfwError(int id, const char* description)
{
	cout << description << endl;
}

int main(const int argc, const char** argv)
{
	string strin("out");
	if (argc > 1) strin = argv[1]; // specify N as the first argument
	glfwSetErrorCallback(&glfwError);
	// OpenGL graphic interface initialization (GLFW + GLEW)
	if (!glfwInit())
	{
		cerr << "Error: Cannot initialize GLFW." << endl;
		return -1;
	}
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	const int l = 792, h = 792;

	GLFWwindow* window = glfwCreateWindow(l, h, "Particle Viewer", nullptr, nullptr);
	// glfwGetPrimaryMonitor() for fullscreen (as 4th argument)

	if (!window) {
		cerr << "GLFW Error: Cannot create window." << endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	if (Init(l, h) == -1)
	{
		Terminate();
		return -1;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	char* c_buf = nullptr;
	int nBodies = 0, old_bytes = 0, iter = 0;

	while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS)
	{
		auto begin = chrono::steady_clock::now();
		glfwPollEvents();

		ifstream fin(strin + "/out" + to_string(iter*20) + '_' + std::to_string(0.005) + ".bin",
					 ios::in | ios::binary);
		int bytes = 0;
		
		if (fin)
		{
			fin.ignore(numeric_limits<streamsize>::max());
			bytes = (int)fin.gcount();
			if (bytes > old_bytes)
			{
				if (c_buf != nullptr)
					delete[] c_buf;
				c_buf = new char[bytes];
				old_bytes = bytes;
			}
			fin.clear();
			fin.seekg(0, ios::beg);
			fin.read(c_buf, bytes);
		}
		else
		{
			clog << "Iteration " << iter << " does not have an associated input file." << endl;
			return 0;
		}
		fin.close();
		float* buf = (float*)c_buf;
		double *d_buf = (double*)c_buf;
		for (int i = 0; i < bytes / 2 / sizeof(double); ++i)
			buf[i] = (float)d_buf[i] * 10e4 * 250.f; // Rescaling to fit the window // window side = 2*4 mm = 8 mm
		nBodies = bytes / 4 / sizeof(double);
		//cout << nBodies << endl;

		glGenBuffers(1, &vb);
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*nBodies*2, buf, GL_DYNAMIC_DRAW);

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(progID);
		
		glBindVertexArray(va);
		glBindBuffer(GL_ARRAY_BUFFER, vb);

		glVertexAttribPointer(
			0,			// attribute 0
			2,			// number of coordinates per vertex
			GL_FLOAT,	// type
			GL_FALSE,	// normalized?
			0,			// stride
			(void*)0	// array buffer offset
		);
		glDrawArrays(GL_POINTS, 0, nBodies);

		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);

		font->Begin();

		font->Color(0, 1, 0, 1);
		{
			font->Draw(
				std::to_string(iter),
				24, 24, 1
			);
		}
		font->End();

		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);

		BYTE* pixels = new BYTE[3 * l * h];

		glReadPixels(0, 0, l, h, GL_RGB, GL_UNSIGNED_BYTE, pixels);

		string imname("img/image");
		imname += to_string(iter);
		imname += ".bmp";

		// Convert to FreeImage format & save to file
		FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, l, h, 3 * l, 24,
			0x0000FF, 0x00FF00, 0xFF0000, false);
		FreeImage_Save(FIF_BMP, image, imname.c_str(), 0);

		// Free resources
		FreeImage_Unload(image);
		delete [] pixels;
		
		glfwSwapBuffers(window);

		GLenum err = glGetError();
		if (err != GL_NO_ERROR)
		{
			cerr << "Drawing error! Error code: " << err << endl;
			return err;
		}
		
		auto end = chrono::steady_clock::now();
		auto milli = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		if (milli < 50)
			this_thread::sleep_for(chrono::milliseconds(50-milli)); // 20 FPS
		++iter;
	}

	Terminate();
	return 0;
}

GLuint LoadShader(const char* VertexSPath, const char* FragmentSPath)
{
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);

	std::string VertexSCode;
	std::ifstream vsstream(VertexSPath, std::ios::in);
	if (vsstream.is_open())
	{
		std::stringstream sstr;
		sstr << vsstream.rdbuf();
		VertexSCode = sstr.str();
		vsstream.close();
	}
	else
	{
		cerr << "Cannot open " << FragmentSPath << '.' << endl;
		return 0;
	}

	std::string FragmentSCode;
	std::ifstream fsstream(FragmentSPath, std::ios::in);
	if (fsstream.is_open())
	{
		std::stringstream sstr;
		sstr << fsstream.rdbuf();
		FragmentSCode = sstr.str();
		fsstream.close();
	}
	else
	{
		cerr << "Cannot open " << FragmentSPath << '.' << endl;
		return 0;
	}

	GLint st;
	int n;

	const char *pS = VertexSCode.c_str();
	glShaderSource(vs, 1, &pS, nullptr);
	glCompileShader(vs);

	glGetShaderiv(vs, GL_COMPILE_STATUS, &st);
	glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &n);
	if (n > 0)
	{
		char* mess = new char[n + 1];
		glGetShaderInfoLog(vs, n, NULL, mess);
		clog << mess << endl;
		delete[] mess;
	}

	pS = FragmentSCode.c_str();

	glShaderSource(fs, 1, &pS, nullptr);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &st);
	glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &n);
	if (n > 0)
	{
		char* mess = new char[n + 1];
		glGetShaderInfoLog(fs, n, NULL, mess);
		clog << mess << endl;
		delete[] mess;
	}

	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);

	glGetProgramiv(prog, GL_LINK_STATUS, &st);
	glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &n);
	if (n > 0)
	{
		char* mess = new char[n + 1];
		glGetProgramInfoLog(prog, n + 1, NULL, mess);
		clog << mess << endl;
		delete[] mess;
	}

	glDetachShader(prog, vs);
	glDetachShader(prog, fs);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return prog;
}