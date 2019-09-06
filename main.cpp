#include "Particle.h"

int main(){

	Particle particle;

	do {
		particle.update(0.001);
		particle.info();
	} while(particle.getZ() > 0.0); // until particle has landed

}
