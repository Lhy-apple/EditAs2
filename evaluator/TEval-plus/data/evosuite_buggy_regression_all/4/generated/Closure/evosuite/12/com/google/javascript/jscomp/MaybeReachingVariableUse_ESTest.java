/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:08:27 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.MaybeReachingVariableUse;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.rhino.Node;
import java.io.ByteArrayOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MaybeReachingVariableUse_ESTest extends MaybeReachingVariableUse_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = compiler0.parseTestCode("+.");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, true);
      Scope scope0 = new Scope(node0, compiler0);
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.getUses("]Lvx{D#", node0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Graph initialized with node annotations turned off
         //
         verifyException("com.google.javascript.jscomp.graph.LinkedDirectedGraph$LinkedDirectedGraphNode", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = compiler0.parseTestCode("+.");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      maybeReachingVariableUse0.analyze();
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = compiler0.parseTestCode("+.");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      assertNotNull(maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      mockPrintStream0.println((Object) maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      SourceFile sourceFile0 = SourceFile.fromCode("+.", "Ao>z4E?7w", "");
      boolean boolean0 = maybeReachingVariableUse_ReachingUses0.equals(sourceFile0);
      assertFalse(boolean0);
  }
}