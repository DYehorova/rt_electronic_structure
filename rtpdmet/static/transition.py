import time
import rt_electronic_structure.rtpdmet.dynamics.system_mod as system_mod
import rt_electronic_structure.rtpdmet.dynamics.fragment_mod as fragment_mod_dynamic


def transition(
        the_dmet, Nsites, Nele, Nfrag, impindx, h_site,
        V_site, hamtype, hubsite_indx, periodic):
    transition_time = time.time()
    mf1RDM = the_dmet.mf1RDM
    tot_system = system_mod.system(
        Nsites, Nele, Nfrag, impindx, h_site, V_site,
        hamtype, mf1RDM, hubsite_indx, periodic)
    tot_system.glob1RDM = the_dmet.glob1RDM
    tot_system.mf1RDM = the_dmet.mf1RDM
    tot_system.NOevecs = the_dmet.NOevecs
    tot_system.NOevals = the_dmet.NOevals
    tot_system.frag_list = []
    for i in range(Nfrag):
        tot_system.frag_list.append(
            fragment_mod_dynamic.fragment(impindx[i], Nsites, Nele))
        tot_system.frag_list[i].rotmat = the_dmet.frag_list[i].rotmat
        tot_system.frag_list[i].CIcoeffs = the_dmet.frag_list[i].CIcoeffs
    print("time to transfer information from static to dynamics",
          time.time()-transition_time)
    return tot_system
