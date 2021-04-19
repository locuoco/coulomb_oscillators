// VERTEX SHADER
#version 330

// Input (buffer)
layout(location = 0) in vec2 pos;

//uniform mat4 MVP;

void main()
{
	gl_Position = vec4(pos, 0, 1); // MVP * vec4(pos, 0, 1)
}