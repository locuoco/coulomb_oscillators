// FRAGMENT SHADER (FONT)
#version 330

// Input (vertex shader & uniform)
in vec2 texcoord;
uniform sampler2D tex;
uniform vec4 col;

// Output (schermo)
out vec4 color;

void main()
{
	color = vec4(1, 1, 1, texture2D(tex, texcoord).r) * col;
}