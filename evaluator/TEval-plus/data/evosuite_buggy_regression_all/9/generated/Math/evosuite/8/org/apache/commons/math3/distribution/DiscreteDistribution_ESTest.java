/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:35:27 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import org.apache.commons.math3.distribution.DiscreteDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.util.Pair;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DiscreteDistribution_ESTest extends DiscreteDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Double double0 = new Double(2967.3);
      Pair<String, Double> pair0 = new Pair<String, Double>("org.apache.commons.math3.distribution.DiscreteDistribution", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      discreteDistribution0.reseedRandomGenerator((-640L));
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      LinkedList<Pair<Object, Double>> linkedList0 = new LinkedList<Pair<Object, Double>>();
      Double double0 = new Double((-4389.639982404014));
      Pair<Object, Double> pair0 = new Pair<Object, Double>(linkedList0, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Object> discreteDistribution0 = null;
      try {
        discreteDistribution0 = new DiscreteDistribution<Object>(linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -4,389.64 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.distribution.DiscreteDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Double double0 = new Double(0.13616033282420004);
      Pair<Byte, Double> pair0 = new Pair<Byte, Double>((Byte) null, double0);
      LinkedList<Pair<Byte, Double>> linkedList0 = new LinkedList<Pair<Byte, Double>>();
      linkedList0.add(pair0);
      MersenneTwister mersenneTwister0 = new MersenneTwister((-1L));
      DiscreteDistribution<Byte> discreteDistribution0 = new DiscreteDistribution<Byte>(mersenneTwister0, linkedList0);
      double double1 = discreteDistribution0.probability((Byte) null);
      assertEquals(1.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Double double0 = new Double(2049.693985651748);
      Pair<String, Double> pair0 = new Pair<String, Double>("", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      double double1 = discreteDistribution0.probability("NaN is not allowed");
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Double double0 = new Double(2054.4648665656337);
      Pair<String, Double> pair0 = new Pair<String, Double>("NaN is not allowed", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      double double1 = discreteDistribution0.probability((String) null);
      assertEquals(0.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Double double0 = new Double(2049.693985651748);
      Pair<String, Double> pair0 = new Pair<String, Double>("NaN is not allowed", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      double double1 = discreteDistribution0.probability("NaN is not allowed");
      assertEquals(1.0, double1, 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Double double0 = new Double(2967.3);
      Pair<String, Double> pair0 = new Pair<String, Double>("org.apache.commons.math3.distribution.DiscreteDistribution", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      List<Pair<String, Double>> list0 = discreteDistribution0.getSamples();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Double double0 = new Double(2967.3);
      Pair<String, Double> pair0 = new Pair<String, Double>("org.apache.commons.math3.distribution.DiscreteDistribution", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      linkedList0.add(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      String string0 = discreteDistribution0.sample();
      assertEquals("org.apache.commons.math3.distribution.DiscreteDistribution", string0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      LinkedList<Pair<Integer, Double>> linkedList0 = new LinkedList<Pair<Integer, Double>>();
      Integer integer0 = new Integer(1282);
      Double double0 = new Double(1282);
      Pair<Integer, Double> pair0 = new Pair<Integer, Double>(integer0, double0);
      linkedList0.add(pair0);
      DiscreteDistribution<Integer> discreteDistribution0 = new DiscreteDistribution<Integer>(linkedList0);
      Integer[] integerArray0 = discreteDistribution0.sample(1282);
      assertEquals(1282, integerArray0.length);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Double double0 = new Double(2049.693985651748);
      Pair<String, Double> pair0 = new Pair<String, Double>("NaN is not allowed", double0);
      LinkedList<Pair<String, Double>> linkedList0 = new LinkedList<Pair<String, Double>>();
      linkedList0.addFirst(pair0);
      DiscreteDistribution<String> discreteDistribution0 = new DiscreteDistribution<String>(linkedList0);
      try { 
        discreteDistribution0.sample((-545518084));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // number of samples (-545,518,084)
         //
         verifyException("org.apache.commons.math3.distribution.DiscreteDistribution", e);
      }
  }
}
