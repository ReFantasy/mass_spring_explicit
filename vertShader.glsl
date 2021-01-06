#version 430

layout (location=0) in vec3 position;
out vec4 varyingColor;

void main(void)
{
	gl_Position =  vec4(position,1.0);
	varyingColor = vec4(position,1.0)*0.5 + vec4(0.5, 0.5, 0.5, 0.5);
}
