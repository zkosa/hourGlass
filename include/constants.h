#ifndef CONSTANTS_H_
#define CONSTANTS_H_

constexpr float pi = 3.14159265358979323846f;

constexpr float g = 9.81f;

// workaround because of no support for global/static variables in device code
#define GRAVITY Vec3d{0.0f, -g, 0.0f}

// TODO: check the validity of the values also in the view of GPU use!
constexpr float SMALL = 1e-10f;
constexpr float VSMALL = 1e-30f;
constexpr float LARGE = 1e10f;
constexpr float VLARGE = 1e30f;

#endif /* CONSTANTS_H_ */
