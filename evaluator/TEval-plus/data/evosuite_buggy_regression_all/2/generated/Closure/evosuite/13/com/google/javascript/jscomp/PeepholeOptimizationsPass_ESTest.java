/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:25:39 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.AbstractPeepholeOptimization;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PeepholeCollectPropertyAssignments;
import com.google.javascript.jscomp.PeepholeFoldWithTypes;
import com.google.javascript.jscomp.PeepholeOptimizationsPass;
import com.google.javascript.jscomp.PeepholeRemoveDeadCode;
import com.google.javascript.jscomp.PeepholeSimplifyRegExp;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeOptimizationsPass_ESTest extends PeepholeOptimizationsPass_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[8];
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      AbstractCompiler abstractCompiler0 = peepholeOptimizationsPass0.getCompiler();
      assertSame(compiler0, abstractCompiler0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[0];
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "j%26", "j%26");
      peepholeOptimizationsPass0.process(node0, node0);
      peepholeOptimizationsPass0.process(node0, node0);
      assertFalse(node0.isCall());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PeepholeSimplifyRegExp peepholeSimplifyRegExp0 = new PeepholeSimplifyRegExp();
      PeepholeSimplifyRegExp peepholeSimplifyRegExp1 = new PeepholeSimplifyRegExp();
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      PeepholeFoldWithTypes peepholeFoldWithTypes0 = new PeepholeFoldWithTypes();
      PeepholeCollectPropertyAssignments peepholeCollectPropertyAssignments0 = new PeepholeCollectPropertyAssignments();
      PeepholeFoldWithTypes peepholeFoldWithTypes1 = new PeepholeFoldWithTypes();
      PeepholeSimplifyRegExp peepholeSimplifyRegExp2 = new PeepholeSimplifyRegExp();
      PeepholeSimplifyRegExp peepholeSimplifyRegExp3 = new PeepholeSimplifyRegExp();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[7];
      abstractPeepholeOptimizationArray0[0] = (AbstractPeepholeOptimization) peepholeCollectPropertyAssignments0;
      abstractPeepholeOptimizationArray0[1] = (AbstractPeepholeOptimization) peepholeCollectPropertyAssignments0;
      abstractPeepholeOptimizationArray0[2] = (AbstractPeepholeOptimization) peepholeFoldWithTypes1;
      abstractPeepholeOptimizationArray0[3] = (AbstractPeepholeOptimization) peepholeSimplifyRegExp1;
      abstractPeepholeOptimizationArray0[4] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      abstractPeepholeOptimizationArray0[5] = (AbstractPeepholeOptimization) peepholeSimplifyRegExp2;
      abstractPeepholeOptimizationArray0[6] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "JSC_XMODULE_REQUIRE_ERROR", "TcRJ.");
      node0.addChildrenToBack(node0);
      peepholeOptimizationsPass0.process(node0, node0);
      assertFalse(node0.hasMoreThanOneChild());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PeepholeRemoveDeadCode peepholeRemoveDeadCode0 = new PeepholeRemoveDeadCode();
      AbstractPeepholeOptimization[] abstractPeepholeOptimizationArray0 = new AbstractPeepholeOptimization[3];
      abstractPeepholeOptimizationArray0[0] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      abstractPeepholeOptimizationArray0[1] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      abstractPeepholeOptimizationArray0[2] = (AbstractPeepholeOptimization) peepholeRemoveDeadCode0;
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "Uf,s/UBUft", "Uf,s/UBUft");
      PeepholeOptimizationsPass peepholeOptimizationsPass0 = new PeepholeOptimizationsPass(compiler0, abstractPeepholeOptimizationArray0);
      peepholeOptimizationsPass0.process(node0, node0);
      assertFalse(node0.hasChildren());
      assertFalse(node0.hasOneChild());
  }
}