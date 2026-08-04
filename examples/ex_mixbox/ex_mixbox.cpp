#include "highmap.hpp"

int main(void)
{
  const glm::ivec2 shape = {512, 128};

  // Define two classic pigment colors: Ultramarine Blue and Cadmium Yellow
  // Ultramarine Blue: RGB (25, 0, 89) -> normalized: (0.098f, 0.0f, 0.349f)
  // Cadmium Yellow: RGB (254, 236, 0) -> normalized: (0.996f, 0.925f, 0.0f)
  const glm::vec3 blue(0.098f, 0.0f, 0.349f);
  const glm::vec3 yellow(0.996f, 0.925f, 0.0f);

  // Create base texture 1 (Blue) with 100% opacity
  hmap::Texture tex1(shape, 4);
  tex1[0] = hmap::Array(shape, blue.x);
  tex1[1] = hmap::Array(shape, blue.y);
  tex1[2] = hmap::Array(shape, blue.z);
  tex1[3] = hmap::Array(shape, 1.0f);

  // Create base texture 2 (Yellow) with a horizontal gradient in the alpha
  // channel
  hmap::Texture tex2(shape, 4);
  tex2[0] = hmap::Array(shape, yellow.x);
  tex2[1] = hmap::Array(shape, yellow.y);
  tex2[2] = hmap::Array(shape, yellow.z);

  hmap::Array alpha(shape);
  for (int j = 0; j < shape.y; ++j)
  {
    for (int i = 0; i < shape.x; ++i)
    {
      alpha(i, j) = float(i) / float(shape.x - 1);
    }
  }
  tex2[3] = alpha;

  // Mix using the three different methods
  hmap::Texture blended_linear = hmap::mix(tex1,
                                           tex2,
                                           hmap::MixMethod::MM_LINEAR);
  hmap::Texture blended_sqrt = hmap::mix(tex1,
                                         tex2,
                                         hmap::MixMethod::MM_SQRT_AVG);
  hmap::Texture blended_mixbox = hmap::mix(tex1,
                                           tex2,
                                           hmap::MixMethod::MM_MIXBOX);

  // Save the blended gradients
  blended_linear.to_png("ex_mixbox_linear.png");
  blended_sqrt.to_png("ex_mixbox_sqrt.png");
  blended_mixbox.to_png("ex_mixbox_mixbox.png");

  return 0;
}
