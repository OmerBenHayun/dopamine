import expiriments.rubust_dqn_asterix_expiriment.code.graph_creator_from_results.utils as loader
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os


output_graph_path = os.path.join('expiriments','rubust_dqn_asterix_expiriment','results','histogram_plots')
def flat(t : list)->list:
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def save_histogram_plot(title :str ,input_to_graph: dict,show_cvar :bool,cvar_p :float= 0.1 ):
    color_list=['red','green','blue']
    res = []
    names = []
    cvar_list = []
    for alpha_train , results in input_to_graph.items():
        names.append(alpha_train)
        res.append(results)
        if show_cvar and int(len(results)*cvar_p) < 1:
            print("cant show cvar.cvar is too small or results are too small")
            return
        if(show_cvar):
            cvar_worst_p_mean = statistics.mean(sorted(results)[:int(len(results)*cvar_p)])
            cvar_list.append(cvar_worst_p_mean)
        #plt.hist(results,label=alpha_train,align='mid', stacked=True)
    #plt.hist(res, label=names,bins= np.arange(max(flat(res))+1)-0.5,align='mid', histtype='bar')


    #one can choose beteween stepfilled or bar.
    #plt.hist(res, label=names,bins = 15, histtype='stepfilled',alpha=0.3,color=color_list)
    plt.hist(res, label=names,color = color_list,bins = 15, histtype='bar')
    if (show_cvar):
        for (name,mean,color) in zip(names,cvar_list,color_list):
            plt.axvline(x=mean,color=color,linestyle='dashed',linewidth= 1,label='{} cvar of {}'.format(cvar_p,name))

    plt.legend()
    plt.ylabel('reward per episode')
    plt.xlabel('Agent rewards')
    plt.title(title)
    file_name = os.path.join(output_graph_path,title.replace(': ','_'))
    file_name = file_name.replace('. ','_')
    file_name = file_name.replace(' ','_')
    print('yay')
    if show_cvar:
        file_name= file_name+ '_with_cvar'
    else:
        file_name= file_name+ '_without_cvar'
    print('save file' + file_name )
    plt.savefig(file_name,format='png')
    # Save the figure and show
    plt.tight_layout()
    plt.clf()
    #plt.show()
    plt.clf()


def main():
   num_of_episodes  = loader.get_num_of_episodes()
   results = loader.load_results()
   for k,v in results.items():
       save_histogram_plot('Game: Astrix. '+'num of episodes: {}. '.format(num_of_episodes) +k[0]+'. '+k[1] ,v,True)
       save_histogram_plot('Game: Astrix. '+'num of episodes: {}. '.format(num_of_episodes) +k[0]+'. '+k[1] ,v,False)
       """
       save_histogram_plot('Game: Astrix. '+'num of episodes: {}. '.format(num_of_episodes) +k[0]+'. '+k[1] ,{
           'a' : [1,2,3,4,5,5,5,5,5,5,4,3,2,1.5,2,2,2,2,3,3,3,3,5,5,5,5,5,5],
           'c' : [1+3,2+3,3+3,4+3,5+3,5+3,5+3,5+3,5+3,5+3,4+3,3+3,2+3,1.5+3,2+3,2+3,2+3,2+3,3+3,3+3,3+3,3+3,5+3,5+3,5+3,5+3,5+3,5],
           'b': [1+1, 2+1, 3+1, 4+1, 5+1, 5+1, 5+1, 5+1, 5+1, 5+1, 4+1, 3+1, 2+1, 1.5+1, 2+1, 2+1, 2+1, 2+1, 3+1, 3+1, 3+1, 3+1, 5+1, 5+1, 5+1, 5+1, 5+1, 5]

       },False)
      """

if __name__ == '__main__':
    print("start")
    main()
    print("finish")
