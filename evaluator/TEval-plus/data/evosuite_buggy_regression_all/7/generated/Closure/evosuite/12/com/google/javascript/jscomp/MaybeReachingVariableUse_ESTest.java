/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:52:47 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.MaybeReachingVariableUse;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.ObjectType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MaybeReachingVariableUse_ESTest extends MaybeReachingVariableUse_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Node node0 = Node.newString((-38), "NxcH|K*d&|i.jcZ", (-38), (-38));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.getUses("NxcH|K*d&|i.jcZ", node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = Node.newString((-38), "NxcH|K*dj|i.jcZ", (-38), (-38));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createInitialEstimateLattice();
      assertNotNull(maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.MaybeReachingVariableUse", "com.google.javascript.jscomp.MaybeReachingVariableUse");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses1 = maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
      assertNotSame(maybeReachingVariableUse_ReachingUses1, maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      boolean boolean0 = maybeReachingVariableUse_ReachingUses0.equals("Named type with empty name component");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      boolean boolean0 = maybeReachingVariableUse_ReachingUses0.equals(maybeReachingVariableUse_ReachingUses0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = new Node(96, 96, 96);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses1 = maybeReachingVariableUse0.join(maybeReachingVariableUse_ReachingUses0, maybeReachingVariableUse_ReachingUses0);
      assertNotSame(maybeReachingVariableUse_ReachingUses1, maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = new Node(98);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = Node.newString(100, "");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Node node0 = new Node(108);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Node node0 = Node.newString(113, "");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = Node.newString(114, "u8J", 114, 114);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = new Node(115);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = maybeReachingVariableUse0.createEntryLattice();
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = Node.newString(118, "ide^?H0(3;i>L", 118, 118);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // AST should be normalized
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = Node.newString(125, "beforeMainOptimizations", 118, (-2231));
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, true);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses1 = maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
      assertNotSame(maybeReachingVariableUse_ReachingUses1, maybeReachingVariableUse_ReachingUses0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MaybeReachingVariableUse.ReachingUses maybeReachingVariableUse_ReachingUses0 = new MaybeReachingVariableUse.ReachingUses();
      Node node0 = new Node(96, 96, 96);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      MaybeReachingVariableUse maybeReachingVariableUse0 = new MaybeReachingVariableUse(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        maybeReachingVariableUse0.flowThrough(node0, maybeReachingVariableUse_ReachingUses0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MaybeReachingVariableUse", e);
      }
  }
}
