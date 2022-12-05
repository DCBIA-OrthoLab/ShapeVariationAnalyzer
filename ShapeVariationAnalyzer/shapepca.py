#!/usr/bin/env python-real


import sys,os
import shapepcalib as shapca
import numpy as np



def printHelp():
    print('Usage:')
    print('  shapca --csv {CSV file path}|--json {JSON file path} [options]')
    print(' ')
    print('Options:')
    print('  --evaluate \tSpecify how many component needs to be used to compute the compactness, the generalization and the specificity of each model. Warning: This operation may take a long time.')
    print('  --shapeNumber \tSpecify the number of shape randomly generated during specificity computation')
    print('  --save \tSpecify a path to save the PCA model (the file should have .json extension ), the model can then be opened with SlicerSALT in the ShapeVariationAnalyzer module.')
    print('  --saveEvaluation \tSet the JSON file path where the evaluation of the models should be saved.')
    print('  --plot \tPlots will be displayed.')
    print('  --minVar \tSet the minimal explained variance per component to be shown in the plot. default; 1.0(%)')
    print('  -h,--help \tDisplay available command line arguments.')

def main ():
    csv_path = None
    json_path = None
    min_variance = 1
    evaluate = False
    shape_number = 10000
    save_evaluation_path = None 
    save_path = None
    show_plot = False
    
    
    #parse
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--csv':
            if json_path:
                print("Error, shapca take a JSON file OR a CSV file")
                printHelp()
                return
            csv_path = sys.argv[i+1]

        if sys.argv[i] == '--json':
            if csv_path:
                print("Error, shapca take a JSON file OR a CSV file")
                printHelp()
                return 
            json_path = sys.argv[i+1]

        if sys.argv[i] == '--save':
            save_path = sys.argv[i+1]

        if sys.argv[i] == '--saveEvaluation':
            save_evaluation_path = sys.argv[i+1]

        if sys.argv[i] == '--plot':
            show_plot = True

        if sys.argv[i] == '--evaluate':
            evaluate = True
            M=int(sys.argv[i+1])

        if sys.argv[i] == '--shapeNumber':
            shape_number=int(sys.argv[i+1])

        if sys.argv[i] == '--minVar':
            min_variance = float(sys.argv[i+1])

        if sys.argv[i] == '--help' or sys.argv[i] == '-h':
            printHelp()
            return

    if csv_path :
        pca = shapca.pcaExplorer()
        pca.loadCSVFile(csv_path)
        pca.process()
    elif json_path:
        pca = shapca.pcaExplorer()
        pca.load(json_path)
    else:
        print('Error, no file to load ...')
        printHelp()
        return

    if evaluate and M != 0:
        groups=list(pca.getGroups())
        dict_evals=dict()
        for i in range(len(groups)):
            group=groups[i]
            print('Group: ',group)
            print('Computing compactness ...')
            compactness,compactness_error=pca.computeCompactness(M_max=M,group=group)
            print('Computing generalization ...')
            generalization,generalization_error=pca.computeGeneralization(M_max=M,group=group)
            print('Computing specificity ...')
            specificity,specificity_error=pca.computeSpecificity(M_max=M,shape_number=shape_number,group=group)
            print('Group: ',group,' done')
            '''dict_eval={}
            dict_eval['compactness']=compactness
            dict_eval['compactness_error']=compactness_error
            dict_eval['generalization']=generalization
            dict_eval['generalization_error']=generalization_error
            dict_eval['specificity']=specificity
            dict_eval['specificity_error']=specificity_error

            dict_evals[group]=dict_eval'''
            
            '''for group, dict_eval in dict_evals.items():
            print('group',group)
            for key, value in dict_eval.items():
                print(key,value)'''
        if json_path:
            pca.updateJSONFile(json_path)

    if save_evaluation_path:
        pca.saveEvaluation(save_evaluation_path)


    if save_path :
        pca.save(save_path)

    if show_plot :
        import matplotlib.pyplot as plt
        plt.close('all')
        groups=list(pca.getGroups())

        for i in range(len(groups)):
            group=groups[i]
            pca.setCurrentPCAModel(group)

            X_pca = pca.current_pca_model["data_projection"]
            pc1 = X_pca[:,0].flatten()
            pc2 = X_pca[:,1].flatten()

            num_component=pca.getRelativeNumComponent(min_variance/100.0)
            level95=np.ones(num_component+2)*95
            level1=np.ones(num_component+2)
            xlevel=np.arange(num_component+2)

            evr = pca.current_pca_model['explained_variance_ratio'][0:num_component].flatten()*100
            sumevr = np.cumsum(evr)

            x = np.arange(1,num_component+1).flatten()


            plt.figure(i+1,figsize=(12, 6))

            plt.subplot(121)
            plt.plot(pc1, pc2,'*')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            if group == 'All':
                plt.title('Projections for all groups')
            else:
                plt.title('Projections for the group '+str(group))


            plt.subplot(122)
            plt.bar(x, evr, label='Explained variance')
            plt.plot(x, sumevr,'-r*', label='Sum Explained Variance')
            plt.plot(xlevel, level95, label='level 95%')
            plt.plot(xlevel, level1, label='Level 1%')
            plt.legend()
            plt.title('explained variance graph')
            plt.xlabel('Component number')
            plt.ylabel('Explained variance (%)')

        
        if evaluate:

            x = np.arange(1,M+1).flatten()
            plt.figure(0,figsize=(12, 10))

            plt.subplot(221)
            for group, model in pca.dictPCA.items():
                plt.errorbar(x,model['compactness'] ,yerr=model['compactness_error'],label=group)
            plt.legend()
            plt.title('Compactness')

            plt.subplot(222)
            for group, model in pca.dictPCA.items():
                plt.errorbar(x,model['generalization'] ,yerr=model['generalization_error'],label=group)
            plt.legend()
            plt.title('Generalization')

            plt.subplot(223)
            for group, model in pca.dictPCA.items():
                plt.errorbar(x,model['specificity'] ,yerr=model['specificity_error'],label=group)
            plt.legend()
            plt.title('Specificity')

        plt.show()



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print ('ERROR, UNEXPECTED EXCEPTION')
        print (str(e))
        import traceback
        traceback.print_exc()
