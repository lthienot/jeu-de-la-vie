
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>
static unsigned couleur = 0xFFFF00FF; // Yellow

unsigned version = 0;

void first_touch_v3 (void);
void first_touch_v4 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);

void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
  first_touch_v3,
  first_touch_v4,
  NULL,
};

int_func_t compute [] = {
  compute_v0,
  compute_v1,
  compute_v2,
  compute_v3,
  compute_v4,
  compute_v5
};

char *version_name [] = {
  "Séquentielle simple",
  "Séquentielle tuilée",
  "Séquentielle tuilée optimisée",
  "OpenMP",
  "OpenMP zone",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  1,
};

///////////////////////////// Version séquentielle simple


unsigned compute_v0 (unsigned nb_iter)
{

  /* for (unsigned it = 1; it <= nb_iter; it ++) { */
  /*   for (int i = 0; i < DIM; i++) */
  /*     for (int j = 0; j < DIM; j++) */
  /* 	next_img (i, j) = cur_img (j, i); */
    
  /*   swap_images (); */
  /* } */
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      for (int i = 0; i < DIM; i++)
	for (int j = 0; j < DIM; j++)
	  {
	    int count = 0;
	    if(i>0 && j>0 && cur_img(i-1,j-1))
	      count++;
	    if(j>0 && cur_img(i,j-1))
	      count++;
	    if(i<DIM-1 && j>0 && cur_img(i+1,j-1))
	      count++;
	    if(i>0 && cur_img(i-1,j))
	      count++;
	    if(i<DIM-1 && cur_img(i+1,j) )
	      count++;
	    if(i>0 && j<DIM-1 && cur_img(i-1,j+1))
	      count++;
	    if(j<DIM-1 && cur_img(i,j+1))
	      count++;
	    if(i<DIM-1 && j<DIM-1 && cur_img(i+1,j+1) )
	      count++;
	    if (cur_img(i,j))
	      if(count < 2 || count > 3)
		next_img(i,j) = 0;
	      else
		next_img(i,j) = cur_img(i,j);
	    else
	      if (count != 3)
		next_img(i,j) = 0;
	      else
		next_img(i,j) = couleur;
	    if (cur_img(i,j)!=next_img(i,j))
	      stop_it = 0;
	  }
      swap_images();

      if (stop_it)
	return it;
    }
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée

#define GRAIN 32

unsigned compute_v1 (unsigned nb_iter)
{
  int tranche = DIM/GRAIN;

  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
	    for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
	      {
		int count = 0;
		if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
		  count++;
		if(jloc>0 && cur_img(iloc,jloc-1))
		  count++;
		if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
		  count++;
		if(iloc>0 && cur_img(iloc-1,jloc))
		  count++;
		if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
		  count++;
		if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
		  count++;
		if(jloc<DIM-1 && cur_img(iloc,jloc+1))
		  count++;
		if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
		  count++;
		if (cur_img(iloc,jloc))
		  if(count < 2 || count > 3)
		    next_img(iloc,jloc) = 0;
		  else
		    next_img(iloc,jloc) = cur_img(iloc,jloc);
		else
		  if (count != 3)
		    next_img(iloc,jloc) = 0;
		  else
		    next_img(iloc,jloc) = couleur;
		if (cur_img(iloc,jloc)!=next_img(iloc,jloc))
		  stop_it = 0;
	      }
      swap_images();
      if (stop_it)
	return it;
    }
  
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}

///////////////////////////// Version séquentielle tuilée optimisée

int cellule[GRAIN][GRAIN];
int cellule_next[GRAIN][GRAIN];

int** next = &cellule_next;
int ** courant = &cellule;

int init = 0;

unsigned compute_v2 (unsigned nb_iter) //ça marche pas !!!!
{
  int tranche = DIM/GRAIN;

  if(!init)
    for (int y = 0; y < GRAIN; y++)
      for (int z = 0; z < GRAIN; z++)
	{
	  init = 1;
	  cellule[y][z] = 0;
	  cellule_next[y][z] = 0;
	}
  
  for (unsigned it = 1; it <= nb_iter; it ++)
    {
      int stop_it = 1;
      
      for (int i = 0; i < GRAIN; i++)
	for (int j = 0; j < GRAIN; j++)
	  if (courant[i][j] == 0)
	    {
	      if(i>0)
		{
		  next[i-1][j] = 0;
		  if(j>0)
		    next[i-1][j-1] = 0;
		  if(j<GRAIN-1)
		    next[i-1][j+1] = 0;
		}
	      if(i<GRAIN-1)
		{
		  next[i+1][j] = 0;
		  if(j>0)
		    next[i+1][j-1] = 0;
		  if(j<GRAIN-1)
		    next[i+1][j+1] = 0;
		}
	      if (j>0)
		next[i][j-1] = 0;
	      if(j<GRAIN-1)
		next[i][j+1] = 0;


	      int stop_tuile = 1;
	      for(int iloc = i*tranche; iloc < (i+1)*tranche && iloc < DIM; iloc++)
		for(int jloc = j*tranche; jloc < (j+1)*tranche && jloc < DIM; jloc++)
		  {
		    int count = 0;
		    if(iloc>0 && jloc>0 && cur_img(iloc-1,jloc-1))
		      count++;
		    if(jloc>0 && cur_img(iloc,jloc-1))
		      count++;
		    if(iloc<DIM-1 && jloc>0 && cur_img(iloc+1,jloc-1))
		      count++;
		    if(iloc>0 && cur_img(iloc-1,jloc))
		      count++;
		    if(iloc<DIM-1 && cur_img(iloc+1,jloc) )
		      count++;
		    if(iloc>0 && jloc<DIM-1 && cur_img(iloc-1,jloc+1))
		      count++;
		    if(jloc<DIM-1 && cur_img(iloc,jloc+1))
		      count++;
		    if(iloc<DIM-1 && jloc<DIM-1 && cur_img(iloc+1,jloc+1) )
		      count++;
		    if (cur_img(iloc,jloc))
		      if(count < 2 || count > 3)
			next_img(iloc,jloc) = 0;
		      else
			next_img(iloc,jloc) = cur_img(iloc,jloc);
		    else
		      if (count != 3)
			next_img(iloc,jloc) = 0;
		      else
			next_img(iloc,jloc) = couleur;
		    if (cur_img(iloc,jloc) != next_img(iloc,jloc))
		      {
			stop_it = 0;
			stop_tuile = 0;
		      }
		  }
	      next[i][j] = stop_tuile;
	      //printf("cellule : %d\n", stop_tuile);
	    }
      swap_images();

      int ***tmp;
      *tmp = courant;
      *courant = next;
      *next = tmp; //cf sur google pr mettre correctement
      
      if (stop_it)
	return it;
    }
  
  // retourne le nombre d'étapes nécessaires à la
  // stabilisation du calcul ou bien 0 si le calcul n'est pas
  // stabilisé au bout des nb_iter itérations
  return 0;
}


///////////////////////////// Version OpenMP de base

void first_touch_v3 ()
{
  int i,j ;

#pragma omp parallel for
  for(i=0; i<DIM ; i++) {
    for(j=0; j < DIM ; j += 512)
      next_img (i, j) = cur_img (i, j) = 0 ;
  }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3(unsigned nb_iter)
{
  return 0;
}



///////////////////////////// Version OpenMP optimisée

void first_touch_v4 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v4(unsigned nb_iter)
{
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v5 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}
