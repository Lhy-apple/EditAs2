/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:35:19 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.LiveVariablesAnalysis;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.ObjectType;
import java.io.DataOutputStream;
import java.io.OutputStream;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LiveVariablesAnalysis_ESTest extends LiveVariablesAnalysis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Node node0 = Node.newString("", (-2392), (-2392));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      Set<Scope.Var> set0 = liveVariablesAnalysis0.getEscapedLocals();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Node node0 = Node.newString((-2396), "");
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        liveVariablesAnalysis0.getVarIndex("ZqF+");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.LiveVariablesAnalysis", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Node node0 = Node.newString("", (-2359), (-2359));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      liveVariablesAnalysis0.analyze(2);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Node node0 = Node.newString("", (-2392), (-2392));
      Scope scope0 = new Scope(node0, (ObjectType) null);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      MockPrintStream mockPrintStream0 = new MockPrintStream(dataOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createInitialEstimateLattice();
      // Undeclared exception!
      try { 
        liveVariablesAnalysis_LiveVariableLattice0.isLive((Scope.Var) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Node node0 = Node.newString("", (-2392), (-2392));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createEntryLattice();
      String string0 = liveVariablesAnalysis_LiveVariableLattice0.toString();
      assertEquals("{}", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Node node0 = Node.newString("", (-2392), (-2392));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, (AbstractCompiler) null);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createInitialEstimateLattice();
      boolean boolean0 = liveVariablesAnalysis_LiveVariableLattice0.isLive(35);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Node node0 = new Node((-2392));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createEntryLattice();
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice1 = liveVariablesAnalysis0.join(liveVariablesAnalysis_LiveVariableLattice0, liveVariablesAnalysis_LiveVariableLattice0);
      assertTrue(liveVariablesAnalysis_LiveVariableLattice1.equals((Object)liveVariablesAnalysis_LiveVariableLattice0));
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Node node0 = Node.newString("  bQ=p,F[sWp1i7X\"a", (-2404), (-2404));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createEntryLattice();
      boolean boolean0 = liveVariablesAnalysis_LiveVariableLattice0.equals(liveVariablesAnalysis0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Node node0 = Node.newString("", (-2359), (-2359));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      ControlFlowGraph.Branch controlFlowGraph_Branch0 = ControlFlowGraph.Branch.ON_FALSE;
      controlFlowGraph0.connectToImplicitReturn(node0, controlFlowGraph_Branch0);
      Compiler compiler0 = new Compiler();
      LiveVariablesAnalysis liveVariablesAnalysis0 = new LiveVariablesAnalysis(controlFlowGraph0, scope0, compiler0);
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice0 = liveVariablesAnalysis0.createEntryLattice();
      LiveVariablesAnalysis.LiveVariableLattice liveVariablesAnalysis_LiveVariableLattice1 = liveVariablesAnalysis0.flowThrough(node0, liveVariablesAnalysis_LiveVariableLattice0);
      assertNotSame(liveVariablesAnalysis_LiveVariableLattice1, liveVariablesAnalysis_LiveVariableLattice0);
  }
}