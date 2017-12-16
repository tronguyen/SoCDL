#include "ctr.h"
#include "opt.h"

extern gsl_rng * RANDOM_NUMBER;
int min_iter = 1; //15 cw
double beta_smooth = 0.01;

c_ctr::c_ctr() {
  m_delta = NULL;
  m_beta = NULL;
  m_theta = NULL;
  m_U = NULL;
  m_V = NULL;

  m_num_factors = 0; // m_num_topics
  m_num_items = 0; // m_num_docs
  m_num_users = 0; // num of users
}

c_ctr::~c_ctr() {
  // free memory
  if (m_delta != NULL) gsl_matrix_free(m_delta);
  if (m_beta != NULL) gsl_matrix_free(m_beta);
  if (m_theta != NULL) gsl_matrix_free(m_theta);
  if (m_U != NULL) gsl_matrix_free(m_U);
  if (m_V != NULL) gsl_matrix_free(m_V);
}

void c_ctr::read_init_information(const char* delta_init_path,
                                  const char* theta_init_path, 
                                  const char* beta_init_path,
                                  const c_corpus* c,
                                  double alpha_smooth) {
  FILE * f = NULL;
  int num_topics = m_num_factors;
  // OFF
  m_theta = gsl_matrix_alloc(c->m_num_docs, num_topics);
  printf("\nreading theta initialization from %s\n", theta_init_path);
  f = fopen(theta_init_path, "r");
  mtx_fscanf(f, m_theta);
  fclose(f);
  // ON
  // m_theta = gsl_matrix_calloc(m_num_items, m_num_factors); // important, should remove once running

  //smoothing
  gsl_matrix_add_constant(m_theta, alpha_smooth);

  //normalize m_theta, in case it's not
  // for (size_t j = 0; j < m_theta->size1; j ++) {
  //   gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);
  //   vnormalize(&theta_v.vector);
  // }

  m_beta = gsl_matrix_alloc(num_topics, c->m_size_vocab);
  printf("reading beta initialization from %s\n", beta_init_path);
  f = fopen(beta_init_path, "r");
  mtx_fscanf(f, m_beta);
  fclose(f);

  // exponentiate if it's not
  // if (mget(m_beta, 0, 0) < 0) {
  //   mtx_exp(m_beta);
  // }
  // else {
  //   gsl_matrix_add_constant(m_beta, beta_smooth);
  //   for (size_t j = 0; j < m_beta->size1; j ++) {
  //     gsl_vector_view beta_v = gsl_matrix_row(m_beta, j);
  //     vnormalize(&beta_v.vector);
  //   }
  // }
  
  if (delta_init_path!=NULL){
    // my code
    m_delta = gsl_matrix_alloc(m_num_users, m_num_factors);
    printf("\nreading delta initialization from %s\n", delta_init_path);
    f = fopen(delta_init_path, "r");
    mtx_fscanf(f, m_delta);
    fclose(f);
    //normalize m_delta, in case it's not
    // for (size_t j = 0; j < m_delta->size1; j ++) {
    //   gsl_vector_view delta_u = gsl_matrix_row(m_delta, j);
    //   vnormalize(&delta_u.vector);
    // }
  }
}

void c_ctr::set_model_parameters(int num_factors, 
                                 int num_users, 
                                 int num_items) {
  m_num_factors = num_factors;
  m_num_users = num_users;
  m_num_items = num_items;
}

void c_ctr::init_model(int ctr_run, int dae_run) {

  m_U = gsl_matrix_calloc(m_num_users, m_num_factors);
  m_V = gsl_matrix_calloc(m_num_items, m_num_factors);

  if (!ctr_run) {
    // this is for convenience, so that updates are similar.
    m_theta = gsl_matrix_calloc(m_num_items, m_num_factors);

    for (size_t i = 0; i < m_V->size1; i ++) 
      for (size_t j = 0; j < m_V->size2; j ++) 
        mset(m_V, i, j, runiform());
  }

  if(!dae_run){ // should be negative
    m_delta = gsl_matrix_calloc(m_num_users, m_num_factors);
  }

}


void c_ctr::stochastic_learn_map_estimate(const c_data* users, const c_data* items, 
                               const c_corpus* c, const ctr_hyperparameter* param,
                               const char* directory) {
  // init model parameters
  printf("\nrunning stochastic learning ...\n");
  printf("initializing the model ...\n");

}

void c_ctr::learn_map_estimate(const c_data* users, const c_data* items, 
                               const c_corpus* c, const ctr_hyperparameter* param,
                               const char* directory) {
  // init model parameters
  printf("\ninitializing the model ...\n");
  init_model(param->ctr_run, param->dae_run);

  // filename
  char name[500];

  // start time
  time_t start, current;
  time(&start);
  int elapsed = 0;

  int iter = 0;
  double likelihood = -exp(50), likelihood_old, llk_ratings, llk_u, llk_v;
  double converge = 1.0;

  /// create the state log file 
  sprintf(name, "%s/state.log", directory);
  FILE* file = fopen(name, "w");
  fprintf(file, "iter time likelihood:[r:u:v] converge\n");


  /* alloc auxiliary variables */
  gsl_matrix* XX = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* A  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* B  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_vector* x  = gsl_vector_alloc(m_num_factors);

  gsl_matrix* phi = NULL;
  gsl_matrix* word_ss = NULL;
  gsl_matrix* log_beta = NULL;
  gsl_vector* gamma = NULL;

  if (param->ctr_run && param->theta_opt) {
    int max_len = c->max_corpus_length();
    phi = gsl_matrix_calloc(max_len, m_num_factors);
    word_ss = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    log_beta = gsl_matrix_calloc(m_num_factors, c->m_size_vocab);
    gsl_matrix_memcpy(log_beta, m_beta);
    mtx_log(log_beta);
    gamma = gsl_vector_alloc(m_num_factors);
  }

  /* tmp variables for indexes */
  int i, j, m, n, l, k;
  int* item_ids; 
  int* user_ids;

  double result;

  /// confidence parameters
  double a_minus_b = param->a - param->b;

  // My code  
  if(param->dae_run){
    printf("\nreading V initialization from %s/final-V.dat\n", directory);
    sprintf(name, "%s/final-V.dat", directory);
    FILE * file_init_V = fopen(name, "r");
    mtx_fscanf(file_init_V, m_V);
    fclose(file_init_V);


    printf("\nreading U initialization from %s/final-U.dat\n", directory);
    sprintf(name, "%s/final-U.dat", directory);
    FILE * file_init_U = fopen(name, "r");
    mtx_fscanf(file_init_U, m_U);
    fclose(file_init_U);

    
    // sprintf(name, "%s/final-phi.dat", directory);
    // if (file_exists(name)) {
    //   printf("\nreading phi initialization from %s/final-phi.dat\n", directory);
    //   FILE * file_init_phi = fopen(name, "r");
    //   mtx_fscanf(file_init_phi, phi);
    //   fclose(file_init_phi);
    // }
 
    // sprintf(name, "%s/final-gamma.dat", directory);
    // if (file_exists(name)) {
    //   printf("\nreading gamma initialization from %s/final-gamma.dat\n", directory);
    //   FILE * file_init_gamma = fopen(name, "r");
    //   vct_fscanf(file_init_gamma, gamma);
    //   fclose(file_init_gamma);
    // }

    // sprintf(name, "%s/final-wordss.dat", directory);
    // if (file_exists(name)) {
    //   printf("\nreading wordss initialization from %s/final-wordss.dat\n", directory);
    //   FILE * file_init_wordss = fopen(name, "r");
    //   mtx_fscanf(file_init_wordss, word_ss);
    //   fclose(file_init_wordss);
    // }

    // sprintf(name, "%s/final-logbeta.dat", directory);
    // if (file_exists(name)) {
    //   printf("\nreading logbeta initialization from %s/final-logbeta.dat\n", directory);
    //   FILE * file_init_logbeta = fopen(name, "r");
    //   mtx_fscanf(file_init_logbeta, log_beta);
    //   fclose(file_init_logbeta);
    // }

    sprintf(name, "%s/final-likelihood.dat", directory);
    if (file_exists(name)) {
      FILE * file_init_llk = fopen(name, "r");
      fscanf(file_init_llk, "%lf", &likelihood);
      fclose(file_init_llk);
      printf("\nreading old-likelihood = %lf\n", likelihood);
    }

  } 

  while ((iter < param->max_iter and converge > 1e-7 ) or iter < min_iter) {
    // if(iter>0){
    //   printf("\nreading V initialization from %s/ctr-tmp/final-V.dat\n", directory);
    //   sprintf(name, "%s/ctr-tmp/final-V.dat", directory);
    //   FILE * file_init_V = fopen(name, "r");
    //   mtx_fscanf(file_init_V, m_V);
    //   fclose(file_init_V);

    //   printf("\nreading U initialization from %s/ctr-tmp/final-U.dat\n", directory);
    //   sprintf(name, "%s/ctr-tmp/final-U.dat", directory);
    //   FILE * file_init_U = fopen(name, "r");
    //   mtx_fscanf(file_init_U, m_U);
    //   fclose(file_init_U);

    //   printf("\nreading theta initialization from %s/ctr-tmp/final-theta.dat\n", directory);
    //   sprintf(name, "%s/ctr-tmp/final-theta.dat", directory);
    //   FILE * file_init_theta = fopen(name, "r");
    //   mtx_fscanf(file_init_theta, m_theta);
    //   fclose(file_init_theta);

    //   printf("\nreading beta initialization from %s/ctr-tmp/final-beta.dat\n", directory);
    //   sprintf(name, "%s/ctr-tmp/final-beta.dat", directory);
    //   FILE * file_init_beta = fopen(name, "r");
    //   mtx_fscanf(file_init_beta, m_beta);
    //   fclose(file_init_beta);
    // }


    likelihood_old = likelihood;
    likelihood = 0.0;
    llk_ratings = 0.0;
    llk_u = 0.0;
    llk_v = 0.0;

    // update U
    gsl_matrix_set_zero(XX);
    for (j = 0; j < m_num_items; j ++) {
      m = items->m_vec_len[j];
      if (m>0) {
        gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
        gsl_blas_dger(1.0, &v.vector, &v.vector, XX);
      }
    }
    gsl_matrix_scale(XX, param->b);
    // this is only for U
    // gsl_matrix_add_diagonal(XX, param->lambda_u); 

    for (i = 0; i < m_num_users; i ++) {
      gsl_vector_view u = gsl_matrix_row(m_U, i);
      gsl_vector_view delta_u = gsl_matrix_row(m_delta, i);

      item_ids = users->m_vec_data[i];
      n = users->m_vec_len[i];
      if (n > 0) {
        // this user has rated some articles
        gsl_matrix_memcpy(A, XX);
        gsl_vector_set_zero(x);
        for (l=0; l < n; l ++) {
          j = item_ids[l];
          gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
          gsl_blas_dger(a_minus_b, &v.vector, &v.vector, A); 
          gsl_blas_daxpy(param->a, &v.vector, x);
        }

        // My code: update the likelihood
        gsl_blas_daxpy(param->lambda_u, &delta_u.vector, x);
        gsl_matrix_add_diagonal(A, param->lambda_u);  
        matrix_vector_solve(A, x, &u.vector);

        // Mycode: update llk
        // likelihood += -0.5 * mahalanobis_prod(B, &u.vector, &u.vector);
        gsl_vector_memcpy(x, &u.vector);
        gsl_vector_sub(x, &delta_u.vector);
        gsl_blas_ddot(x, x, &result);

        // else{
        //   matrix_vector_solve(A, x, &(u.vector));
        //   gsl_blas_ddot(&u.vector, &u.vector, &result);
        // }

        likelihood += -0.5 * param->lambda_u * result;
        llk_u += -0.5 * param->lambda_u * result;
      }
      else{ // My Code
        if (param->dae_run){
          gsl_vector_memcpy(&u.vector, &delta_u.vector); // update v  
        }
      }
    }

    printf("\nDone update U ...\n");

    // update V
    if (param->ctr_run && param->theta_opt) gsl_matrix_set_zero(word_ss);

    gsl_matrix_set_zero(XX);
    for (i = 0; i < m_num_users; i ++) {
      n = users->m_vec_len[i]; 
      if (n>0) {
        gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);
        gsl_blas_dger(1.0, &u.vector, &u.vector, XX);
      }
    }
    gsl_matrix_scale(XX, param->b);
    // gsl_matrix_add_diagonal(XX, param->lambda_v); 

    for (j = 0; j < m_num_items; j ++) {
      gsl_vector_view v = gsl_matrix_row(m_V, j);
      gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);

      user_ids = items->m_vec_data[j];
      m = items->m_vec_len[j];
      if (m>0) {
        // m > 0, some users have rated this article
        gsl_matrix_memcpy(A, XX);
        gsl_vector_set_zero(x);
        for (l = 0; l < m; l ++) {
          i = user_ids[l];
          gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
          gsl_blas_dger(a_minus_b, &u.vector, &u.vector, A);
          gsl_blas_daxpy(param->a, &u.vector, x);
        }

        // adding the topic vector
        // even when ctr_run=0, m_theta=0
        gsl_blas_daxpy(param->lambda_v, &theta_v.vector, x);
        
        gsl_matrix_memcpy(B, A); // save for computing likelihood 

        // here different from U update
        gsl_matrix_add_diagonal(A, param->lambda_v);  
        matrix_vector_solve(A, x, &v.vector);

        // update the likelihood for the relevant part
        likelihood += -0.5 * m * param->a;

        for (l = 0; l < m; l ++) {
          i = user_ids[l];
          gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
          gsl_blas_ddot(&u.vector, &v.vector, &result);
          likelihood += param->a * result;
          llk_ratings += param->a * result;
        }
        likelihood += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);
        // likelihood part of theta, even when theta=0, which is a
        // special case
        gsl_vector_memcpy(x, &v.vector);
        gsl_vector_sub(x, &theta_v.vector);
        gsl_blas_ddot(x, x, &result);
        likelihood += -0.5 * param->lambda_v * result;
        llk_v += -0.5 * param->lambda_v * result;
        
        if (param->ctr_run && param->theta_opt) {
          const c_document* doc =  c->m_docs[j];
          doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, true); 
          optimize_simplex(gamma, &v.vector, param->lambda_v, &theta_v.vector); 
        }
      }
      else {
      // m=0, this article has never been rated
        if (param->ctr_run && param->theta_opt) {
          // my code
          gsl_vector_memcpy(&v.vector, &theta_v.vector); // update v  
          // CW code
          const c_document* doc =  c->m_docs[j];
          doc_inference(doc, &theta_v.vector, log_beta, phi, gamma, word_ss, false); 
          vnormalize(gamma);
          gsl_vector_memcpy(&theta_v.vector, gamma);
      
        }
      }
    }

    // update beta if needed
    if (param->ctr_run && param->theta_opt) {
        gsl_matrix_memcpy(m_beta, word_ss);
        for (k = 0; k < m_num_factors; k ++) {
          gsl_vector_view row = gsl_matrix_row(m_beta, k);
          vnormalize(&row.vector);
        }
        gsl_matrix_memcpy(log_beta, m_beta);
        mtx_log(log_beta);
    }

    printf("\nDone update V ...\n");

    time(&current);
    elapsed = (int)difftime(current, start);

    iter++;
    converge = fabs((likelihood-likelihood_old)/likelihood_old);

    if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");

    fprintf(file, "%04d %06d %10.5f:[%.5f:%.5f:%.5f] %.10f\n", iter, elapsed, likelihood, llk_ratings, llk_u, llk_v, converge);
    fflush(file);
    printf("iter=%04d, time=%06d, likelihood=%.5f, converge=%.10f\n", iter, elapsed, likelihood, converge);

    // save intermediate results
    if (iter % param->save_lag == 0) {

      sprintf(name, "%s/ctr-tmp/final-U.dat", directory);
      FILE * file_U = fopen(name, "w");
      mtx_fprintf(file_U, m_U);
      fclose(file_U);

      sprintf(name, "%s/ctr-tmp/final-V.dat", directory);
      FILE * file_V = fopen(name, "w");
      mtx_fprintf(file_V, m_V);
      fclose(file_V);

      if (param->ctr_run) { 
        sprintf(name, "%s/ctr-tmp/final-theta.dat", directory);
        FILE * file_theta = fopen(name, "w");
        mtx_fprintf(file_theta, m_theta);
        fclose(file_theta);

        sprintf(name, "%s/ctr-tmp/final-beta.dat", directory);
        FILE * file_beta = fopen(name, "w");
        mtx_fprintf(file_beta, m_beta);
        fclose(file_beta);
      }
    }
    // gsl_matrix_set_zero(XX);
    // gsl_matrix_set_zero(A);
    // gsl_matrix_set_zero(B);
    // gsl_vector_set_zero(x);

    // gsl_matrix_memcpy(log_beta, m_beta);
    // mtx_log(log_beta);
    // gsl_matrix_set_zero(word_ss);
    // gsl_matrix_set_zero(phi);
    // gsl_vector_set_zero(gamma);

  }

  // save final results
  sprintf(name, "%s/final-U.dat", directory);
  FILE * file_U = fopen(name, "w");
  mtx_fprintf(file_U, m_U);
  fclose(file_U);

  sprintf(name, "%s/final-V.dat", directory);
  FILE * file_V = fopen(name, "w");
  mtx_fprintf(file_V, m_V);
  fclose(file_V);

  sprintf(name, "%s/final-likelihood.dat", directory);
  FILE * file_L = fopen(name, "w");
  fprintf(file_L,"%f",likelihood);
  fclose(file_L);

  sprintf(name, "%s/final-likelihood-ruv.dat", directory);
  FILE * file_mL = fopen(name, "w");
  fprintf(file_mL,"%f %f %f",llk_ratings, llk_u, llk_v);
  fclose(file_mL);
  

  if (param->ctr_run) { 
    sprintf(name, "%s/final-theta.dat", directory);
    FILE * file_theta = fopen(name, "w");
    mtx_fprintf(file_theta, m_theta);
    fclose(file_theta);

    sprintf(name, "%s/final-beta.dat", directory);
    FILE * file_beta = fopen(name, "w");
    mtx_fprintf(file_beta, m_beta);
    fclose(file_beta);

    // sprintf(name, "%s/final-phi.dat", directory);
    // FILE * file_phi = fopen(name, "w");
    // mtx_fprintf(file_phi, phi);
    // fclose(file_phi);

    // sprintf(name, "%s/final-gamma.dat", directory);
    // FILE * file_gamma = fopen(name, "w");
    // vct_fprintf(file_gamma, gamma);
    // fclose(file_gamma);

    // sprintf(name, "%s/final-logbeta.dat", directory);
    // FILE * file_logbeta = fopen(name, "w");
    // mtx_fprintf(file_logbeta, log_beta);
    // fclose(file_logbeta);

    // sprintf(name, "%s/final-wordss.dat", directory);
    // FILE * file_wordss = fopen(name, "w");
    // mtx_fprintf(file_wordss, word_ss);
    // fclose(file_wordss);

  }

  // free memory
  gsl_matrix_free(XX);
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_vector_free(x);

  if (param->ctr_run && param->theta_opt) {
    gsl_matrix_free(phi);
    gsl_matrix_free(log_beta);
    gsl_matrix_free(word_ss);
    gsl_vector_free(gamma);
  }
}

double c_ctr::doc_inference(const c_document* doc, const gsl_vector* theta_v, 
                            const gsl_matrix* log_beta, gsl_matrix* phi,
                            gsl_vector* gamma, gsl_matrix* word_ss, 
                            bool update_word_ss) {

  double pseudo_count = 1.0;
  double likelihood = 0;
  gsl_vector* log_theta_v = gsl_vector_alloc(theta_v->size);
  gsl_vector_memcpy(log_theta_v, theta_v);
  vct_log(log_theta_v);

  int n, k, w;
  double x;
  for (n = 0; n < doc->m_length; n ++) {
    w = doc->m_words[n]; 
    for (k = 0; k < m_num_factors; k ++)
      mset(phi, n, k, vget(theta_v, k) * mget(m_beta, k, w));

    gsl_vector_view row =  gsl_matrix_row(phi, n);
    vnormalize(&row.vector);

    for (k = 0; k < m_num_factors; k ++) {
      x = mget(phi, n, k);
      if (x > 0) 
        likelihood += x*(vget(log_theta_v, k) + mget(log_beta, k, w) - log(x));
    }
  }

  if (pseudo_count > 0) {
    likelihood += pseudo_count * vsum(log_theta_v);
  }

  gsl_vector_set_all(gamma, pseudo_count); // smoothing with small pseudo counts
  for (n = 0; n < doc->m_length; n ++) {
    for (k = 0; k < m_num_factors; k ++) {
      x = doc->m_counts[n] * mget(phi, n, k);
      vinc(gamma, k, x);      
      if (update_word_ss) minc(word_ss, k, doc->m_words[n], x);
    }
  }

  gsl_vector_free(log_theta_v);
  return likelihood;
}
