/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:43:58 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.InlineCostEstimator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InlineCostEstimator_ESTest extends InlineCostEstimator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Node node0 = Node.newString(38, "hDe#5~_r/", 32, 18);
      int int0 = InlineCostEstimator.getCost(node0);
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Node node0 = Node.newNumber(1865.5);
      int int0 = InlineCostEstimator.getCost(node0, (-42));
      assertEquals(6, int0);
  }
}
