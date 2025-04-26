package weka.classifiers.gf;


import weka.classifiers.*;
import weka.core.*;

import java.util.*;

import weka.classifiers.bayes.NaiveBayes;

/**
 * Implement an VDM classifier.
 */
public class HISCDM extends AbstractClassifier implements AdditionalMeasureProducer {

  /** The training instances used for classification. */
  private Instances m_Train;

  /** The best number of neighbours to use for classification. */
  private int m_kNN;

  /** define a classifier of NB*/
  private NaiveBayes m_NB;
  
  /** The number of each class value occurs in the dataset */
  private double [] m_ClassCounts;
  
  /** The number of class and each attribute value occurs in the dataset */
  private double [][] m_ClassAttCounts;
  
  /** The number of two attributes values occurs in the dataset */
  private double [][] m_AttAttCounts;

  /** The number of class and two attributes values occurs in the dataset */
  private double [][][] m_ClassAttAttCounts;
  
  /** The number of each attribute value occurs in the dataset */
  private double [] m_AttCounts;

  /** The number of values for each attribute in the dataset */
  private int [] m_NumAttValues;

  /** The starting index of each attribute in the dataset */
  private int [] m_StartAttIndex;

  /** The number of values for all attributes in the dataset */
  private int m_TotalAttValues;

  /** The number of classes in the dataset */
  private int m_NumClasses;

  /** The number of attributes including class in the dataset */
  private int m_NumAttributes;

  /** The number of instances in the dataset */
  private int m_NumInstances;

  /** The index of the class attribute in the dataset */
  private int m_ClassIndex;
  
  /** The 2D array of conditional mutual information of each pair attributes */
  private double[][] m_condiMutualInfo;

  /**
   * Builds IBK classifier.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    //initial data
    m_Train=new Instances(data);
    count(m_Train);
    m_kNN=10; 
  
    }

private void count (Instances instances) throws Exception {

	  m_NumClasses = instances.numClasses();
	  m_ClassIndex = instances.classIndex();
	  m_NumAttributes = instances.numAttributes();
	  m_NumInstances = instances.numInstances();
	  m_TotalAttValues = 0;

	  // allocate space for attribute reference arrays
	  m_StartAttIndex = new int[m_NumAttributes];
	  m_NumAttValues = new int[m_NumAttributes];

	  // set the starting index of each attribute and the number of values for
	  // each attribute and the total number of values for all attributes (not including class).
	  for(int i = 0; i < m_NumAttributes; i++) {
	    if(i != m_ClassIndex) {
	      m_StartAttIndex[i] = m_TotalAttValues;
	      m_NumAttValues[i] = instances.attribute(i).numValues();
	      m_TotalAttValues += m_NumAttValues[i];
	    }
	    else {
	      m_StartAttIndex[i] = -1;
	      m_NumAttValues[i] = m_NumClasses;
	    }
	  }

	  // allocate space for counts and frequencies
	  m_ClassCounts = new double[m_NumClasses];
	  //m_AttCounts = new double[m_TotalAttValues];
	 // m_AttAttCounts = new double[m_TotalAttValues][m_TotalAttValues];
	  m_ClassAttAttCounts = new double[m_NumClasses][m_TotalAttValues][m_TotalAttValues];
	  m_ClassAttCounts = new double[m_NumClasses][m_TotalAttValues];

	  // Calculate the counts
	  for(int k = 0; k < m_NumInstances; k++) {
		    int classVal=(int)instances.instance(k).classValue();
		    m_ClassCounts[classVal] ++;
		    int[] attIndex = new int[m_NumAttributes];
		    for(int i = 0; i < m_NumAttributes; i++) {
		        if(i == m_ClassIndex)
		           attIndex[i] = -1;  // we don't use the class attribute in counts
		           else
		              attIndex[i] = m_StartAttIndex[i] + (int)instances.instance(k).value(i);
		        
		     }

		    for(int Att1 = 0; Att1 < m_NumAttributes; Att1++) {
		      if(attIndex[Att1] == -1 || instances.instance(k).isMissing(Att1)) continue;
		      m_ClassAttCounts[classVal][attIndex[Att1]]++;
		      for(int Att2 = 0; Att2 < m_NumAttributes; Att2++) {
		        if((attIndex[Att2] == -1) || instances.instance(k).isMissing(Att2))continue; 
		         // m_AttAttCounts[attIndex[Att1]][attIndex[Att2]] ++;	
		          m_ClassAttAttCounts[classVal][attIndex[Att1]][attIndex[Att2]] ++;
		      }
		    }
		  }
	//compute conditional mutual information of each pair attributes (not including class)
	  m_condiMutualInfo=new double[m_NumAttributes][m_NumAttributes];
	    for(int son=0;son<m_NumAttributes;son++){
	      if(son == m_ClassIndex) continue;
	      for(int parent=0;parent<m_NumAttributes;parent++){
	        if(parent == m_ClassIndex || son==parent) continue;
	        m_condiMutualInfo[son][parent]=conditionalMutualInfo(son,parent);
	      }
	  }
  }

  /**
   * Computes class distribution for instance.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   */
  public double[] distributionForInstance(Instance instance) throws Exception {

    NeighborList neighborlist = findNeighbors(instance,m_kNN);
    return computeDistribution2(neighborInstances(neighborlist),instance);
  }

  /**
   * Build the list of nearest k neighbors to the given test instance.
   *
   * @param instance the instance to search for neighbours
   * @return a list of neighbors
   */
   private NeighborList findNeighbors(Instance instance,int kNN) {

    double distance;
    NeighborList neighborlist = new NeighborList(kNN);
    for(int i=0; i<m_Train.numInstances();i++){
      Instance trainInstance=m_Train.instance(i);
      distance=distance(instance,trainInstance);
       if (neighborlist.isEmpty()||i<kNN||distance<=neighborlist.m_Last.m_Distance) {
         neighborlist.insertSorted(distance,trainInstance);
       }
    }
    return neighborlist;

  }

  /**
   * Turn the list of nearest neighbors into a probability distribution
   *
   * @param neighborlist the list of nearest neighboring instances
   * @return the probability distribution
   */
  private Instances neighborInstances (NeighborList neighborlist) throws Exception {

    Instances neighborInsts = new Instances(m_Train, neighborlist.currentLength());
    if (!neighborlist.isEmpty()) {
      NeighborNode current = neighborlist.m_First;
      while (current != null) {
        neighborInsts.add(current.m_Instance);
        current = current.m_Next;
      }
    }
    return neighborInsts;

  }

  /**
   * Calculates the distance between two instances
   *
   * @param first the first instance
   * @param second the second instance
   * @return the distance between the two given instances
   */
  private double distance(Instance first, Instance second) {

    // store first's att values in an int array
    int[] attIndexf = new int[m_NumAttributes];
    for(int att = 0; att < m_NumAttributes; att++) {
      if(att == m_ClassIndex)
        attIndexf[att] = -1;
      else
        attIndexf[att] = m_StartAttIndex[att] + (int)first.value(att);
    }
    // store second's att values in an int array
    int[] attIndexs = new int[m_NumAttributes];
    for(int att = 0; att < m_NumAttributes; att++) {
      if(att == m_ClassIndex)
        attIndexs[att] = -1;
      else
        attIndexs[att] = m_StartAttIndex[att] + (int)second.value(att);
    }
    
    int classVal_train = (int)second.classValue();
    
    // calculate the distance between first and second
    double distance = 0;  
    
    for(int att2 = 0; att2 < m_NumAttributes; att2++ ){    
        if(attIndexf[att2]==-1 || attIndexs[att2]==-1) continue;
        if(first.isMissing(att2)){
        	distance+=1;
    	}
        else if((int)first.value(att2)==(int)second.value(att2)) {
        	distance+=0;
        }
        else {
        	double tempprob = 0;
        	double condiMutualInfoSum=0;
        	for(int att1 = 0; att1 < m_NumAttributes; att1++ ){
        		if(att1==m_ClassIndex)continue;
        		if(att2==att1||m_ClassAttAttCounts[classVal_train][attIndexf[att1]][attIndexf[att1]] == 0) {
        			distance+=Math.pow(1-(m_ClassAttAttCounts[classVal_train][attIndexf[att2]][attIndexf[att2]]/m_ClassCounts[classVal_train]), 2);    	    		 
        		}
        		else {
        			condiMutualInfoSum+=m_condiMutualInfo[att2][att1];
        			tempprob += m_condiMutualInfo[att2][att1]*(m_ClassAttAttCounts[classVal_train][attIndexf[att1]][attIndexf[att2]])/(m_ClassAttAttCounts[classVal_train][attIndexf[att1]][attIndexf[att1]]);		
        		}	
        	}
        	if(condiMutualInfoSum>0) {
        		tempprob = tempprob/condiMutualInfoSum;
        		distance+=Math.pow((1-tempprob), 2); 
        	}
        	else {
        		distance+=Math.pow(1-(m_ClassAttCounts[classVal_train][attIndexf[att2]]/m_ClassCounts[classVal_train]), 2);    	    		 
        	}
        } 
    }
    distance=Math.sqrt(distance);
    return distance;
  }

  /**
   * Computes conditional mutual information between a pair of attributes.
   */
  
  private double conditionalMutualInfo(int son, int parent) throws Exception{

	    double CondiMutualInfo=0;
	    int sIndex=m_StartAttIndex[son];
	    int pIndex=m_StartAttIndex[parent];
	    double[] PriorsClass = new double[m_NumClasses];
	    double[][] PriorsClassSon=new double[m_NumClasses][m_NumAttValues[son]];
	    double[][] PriorsClassParent=new double[m_NumClasses][m_NumAttValues[parent]];
	    double[][][] PriorsClassParentSon=new double[m_NumClasses][m_NumAttValues[parent]][m_NumAttValues[son]];

	    for(int i=0;i<m_NumClasses;i++){
	      PriorsClass[i]=m_ClassCounts[i]/m_NumInstances;
	    }

	    for(int i=0;i<m_NumClasses;i++){
	      for(int j=0;j<m_NumAttValues[son];j++){
	        PriorsClassSon[i][j]=m_ClassAttAttCounts[i][sIndex+j][sIndex+j]/m_NumInstances;
	      }
	    }

	    for(int i=0;i<m_NumClasses;i++){
	      for(int j=0;j<m_NumAttValues[parent];j++){
	        PriorsClassParent[i][j]=m_ClassAttAttCounts[i][pIndex+j][pIndex+j]/m_NumInstances;
	      }
	    }

	    for(int i=0;i<m_NumClasses;i++){
	      for(int j=0;j<m_NumAttValues[parent];j++){
	        for(int k=0;k<m_NumAttValues[son];k++){
	          PriorsClassParentSon[i][j][k]=m_ClassAttAttCounts[i][pIndex+j][sIndex+k]/m_NumInstances;
	        }
	      }
	    }

	    for(int i=0;i<m_NumClasses;i++){
	      for(int j=0;j<m_NumAttValues[parent];j++){
	        for(int k=0;k<m_NumAttValues[son];k++){
	          CondiMutualInfo+=PriorsClassParentSon[i][j][k]*log2(PriorsClassParentSon[i][j][k]*PriorsClass[i],PriorsClassParent[i][j]*PriorsClassSon[i][k]);
	        }
	      }
	    }
	    return CondiMutualInfo;
	  }
  
  /**
   * compute the logarithm whose base is 2.
   *
   * @param x numerator of the fraction.
   * @param y denominator of the fraction.
   * @return the natual logarithm of this fraction.
   */
  private double log2(double x,double y){

    if(x<1e-6||y<1e-6)
      return 0.0;
    else
      return Math.log(x/y)/Math.log(2);
  }
  /**
   * Compute the distribution.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  private double[] computeDistribution1(Instances data,Instance instance) throws Exception {

	  int numClasses=data.numClasses();
	    double[] probs=new double[numClasses];
	    double[] classCounts=new double[numClasses];
	    int numInstances=data.numInstances();
	    for (int i=0;i<numInstances;i++){
	      int classVal=(int)data.instance(i).classValue();
	      classCounts[classVal] ++;
	    }
	    for (int i=0;i<numClasses;i++){
	      probs[i]=(classCounts[i]+1.0)/(numInstances+numClasses);
	    }
	    return probs;
	  }

  /**
   * Compute the distribution.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  private double[] computeDistribution2(Instances data,Instance instance) throws Exception {

    int numClasses=data.numClasses();
    int numInstances=data.numInstances();
    double[] probs=new double[numClasses];
    double[] classCounts=new double[numClasses];
    double [] dist=new double[numInstances];
    double [] weight=new double[numInstances];
    Instance inst;
    for (int i=0;i<numInstances;i++){
      inst=data.instance(i);
      int classVal=(int)inst.classValue();
      dist[i]=distance(instance,inst);
      weight[i]=1.0/(1.0+dist[i]*dist[i]);
      classCounts[classVal] +=weight[i];
    }
    double sum=Utils.sum(weight);
    for (int i=0;i<numClasses;i++){
      probs[i]=(classCounts[i]+1.0)/(sum+numClasses);
    }
    return probs;
  }

  /**
   * Compute the distribution.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  private double[] computeDistribution3(Instances data,Instance instance) throws Exception {

	Instances instances=new Instances(data);
	double dist,weight;
	for(int i=0;i<instances.numInstances();i++){
		dist=distance(instance,instances.instance(i));
		weight=1.0/(1.0+dist*dist);
		instances.instance(i).setWeight(weight);
	}
	m_NB=new NaiveBayes();
	m_NB.buildClassifier(instances);
	return m_NB.distributionForInstance(instance);
  }
  
  /**
   * Compute the distribution.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  private double[] computeDistribution4(Instances data,Instance instance) throws Exception {
    // Calculate the frequencies
    int numClasses=data.numClasses();
    int numAttributes=data.numAttributes();
    int numInstances=data.numInstances();
    double [] classCounts=new double[numClasses];
    double [] dist=new double[numInstances];
    double [] weight=new double[numInstances];
    double [][] classAttCounts=new double[numClasses][numAttributes];
    Instance inst;
    for(int i = 0; i < numInstances; i++) {
      inst=data.instance(i);
      dist[i]=distance(instance,inst);
    }
    double dmax=dist[Utils.maxIndex(dist)];
    for(int i = 0; i < numInstances; i++) {
      weight[i]=1-dist[i]/dmax;
    }
    for(int i = 0; i < numInstances; i++) {
      inst=data.instance(i);
      int classVal=(int)inst.classValue();
      classCounts[classVal] +=weight[i];
      for(int j = 0; j < numAttributes; j++) {
        if(j == data.classIndex()) continue;
        if((int)inst.value(j)==(int)instance.value(j))classAttCounts[classVal][j]+=weight[i];
      }
    }
    // calculate probabilities for each possible class value
    double [] probs = new double[numClasses];
    for(int classVal=0;classVal<numClasses;classVal++) {
      probs[classVal]=(classCounts[classVal]+1.0)/(Utils.sum(weight)+numClasses);
      for(int j = 0; j < numAttributes; j++) {
        if(j == data.classIndex()) continue;
        probs[classVal]*=(classAttCounts[classVal][j]+1.0)/(classCounts[classVal]+data.attribute(j).numValues());
      }
    }
    Utils.normalize(probs);
    return probs;
  }

  /**
   * Additional measure --- number of attributes selected
   * @return the number of attributes selected
   */
  public double measureNumAttributesSelected() {
    return m_NumAttributes-1;
  }

  /**
   * Returns an enumeration of the additional measure names
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    Vector newVector = new Vector(1);
    newVector.addElement("measureNumAttributesSelected");
    return newVector.elements();
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @exception IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.compareToIgnoreCase("measureNumAttributesSelected") == 0) {
      return measureNumAttributesSelected();
    }  else {
      throw new IllegalArgumentException(additionalMeasureName
                          + " not supported (AttributeSelectedClassifier)");
    }
  }

  /**
   * Returns the number of the selected attributes.
   *
   * @return a description of the classifier as a string.
   */
  public String toString() {
    StringBuffer text = new StringBuffer();
    text.append("------------\n");
    text.append("The number of the selected attributes:"+measureNumAttributesSelected()+".\n");
    text.append("------------\n");
    return text.toString();
  }

  /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new HISCDM(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

  /*
   * A class for storing data about a neighboring instance
   */
  private class NeighborNode {

    /** The neighbor instance */
    private Instance m_Instance;

    /** The distance from the current instance to this neighbor */
    private double m_Distance;

    /** A link to the next neighbor instance */
    private NeighborNode m_Next;

    /**
     * Create a new neighbor node.
     *
     * @param distance the distance to the neighbor
     * @param instance the neighbor instance
     * @param next the next neighbor node
     */
    public NeighborNode(double distance, Instance instance, NeighborNode next){
      m_Distance = distance;
      m_Instance = instance;
      m_Next = next;
    }

    /**
     * Create a new neighbor node that doesn't link to any other nodes.
     *
     * @param distance the distance to the neighbor
     * @param instance the neighbor instance
     */
    public NeighborNode(double distance, Instance instance) {

      this(distance, instance, null);
    }
  }

  /*
   * A class for a linked list to store the nearest k neighbours to an instance.
   */
  private class NeighborList {

    /** The first node in the list */
    private NeighborNode m_First;

    /** The last node in the list */
    private NeighborNode m_Last;

    /** The number of nodes to attempt to maintain in the list */
    private int m_Length = 1;

    /**
     * Creates the neighborlist with a desired length
     *
     * @param length the length of list to attempt to maintain
     */
    public NeighborList(int length) {

      m_Length = length;
    }

    /**
     * Gets whether the list is empty.
     *
     * @return true if so
     */
    public boolean isEmpty() {

      return (m_First == null);
    }

    /**
     * Gets the current length of the list.
     *
     * @return the current length of the list
     */
    public int currentLength() {

      int i = 0;
      NeighborNode current = m_First;
      while (current != null) {
        i++;
        current = current.m_Next;
      }
      return i;
    }

    /**
     * Inserts an instance neighbor into the list, maintaining the list sorted by distance.
     *
     * @param distance the distance to the instance
     * @param instance the neighboring instance
     */
    public void insertSorted(double distance, Instance instance) {

      if (isEmpty()) {
        m_First = m_Last = new NeighborNode(distance, instance);
      } else {
        NeighborNode current = m_First;
        if (distance < m_First.m_Distance) {// Insert at head
          m_First = new NeighborNode(distance, instance, m_First);
        }
        else { // Insert further down the list
          for( ;(current.m_Next != null) &&
                 (current.m_Next.m_Distance < distance);
               current = current.m_Next);
          current.m_Next = new NeighborNode(distance, instance,
                                            current.m_Next);
          if (current.equals(m_Last)) {
            m_Last = current.m_Next;
          }
        }

        // Trip down the list until we've got k list elements (or more if the distance to the last elements is the same).
        int valcount = 0;
        for(current = m_First; current.m_Next != null;
            current = current.m_Next) {
          valcount++;
          if ((valcount >= m_Length) && (current.m_Distance != current.m_Next.m_Distance)) {
            m_Last = current;
            current.m_Next = null;
            break;
          }
        }
      }
    }

    /**
     * Prunes the list to contain the k nearest neighbors. If there are multiple neighbors at the k'th distance, all will be kept.
     *
     * @param k the number of neighbors to keep in the list.
     */
    public void pruneToK(int k) {

      if (isEmpty()) {
        return;
      }
      if (k < 1) {
        k = 1;
      }
      int currentK = 0;
      double currentDist = m_First.m_Distance;
      NeighborNode current = m_First;
      for(; current.m_Next != null; current = current.m_Next) {
        currentK++;
        currentDist = current.m_Distance;
        if ((currentK >= k) && (currentDist != current.m_Next.m_Distance)) {
          m_Last = current;
          current.m_Next = null;
          break;
        }
      }
    }

  }

}



