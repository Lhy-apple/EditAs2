/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:00:37 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowAnalysis;
import com.google.javascript.jscomp.LineNumberCheck;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.ReferenceCollectingCallback;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ReferenceCollectingCallback_ESTest extends ReferenceCollectingCallback_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node node0 = Node.newString("f^V`F(Fo$Z8A", (-1140), (-1140));
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.getAssignedValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(101, 101, 108);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      Scope scope0 = referenceCollectingCallback_Reference0.getScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(101, 101, 108);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      Node node1 = referenceCollectingCallback_Reference0.getParent();
      assertEquals(42, Node.NO_SIDE_EFFECTS_CALL);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, 118, 118);
      LineNumberCheck lineNumberCheck0 = new LineNumberCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, lineNumberCheck0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[0];
      Node node0 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, referenceCollectingCallback_BasicBlock0, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = referenceCollectingCallback_Reference0.getBasicBlock();
      assertSame(referenceCollectingCallback_BasicBlock0, referenceCollectingCallback_BasicBlock1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node node0 = Node.newNumber(0.0, 101, 101);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      String string0 = referenceCollectingCallback_Reference0.getSourceName();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(101, 101, 108);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      Node node1 = referenceCollectingCallback_Reference0.getGrandparent();
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(120, 120, 120);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isHoistedFunction();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null, (NodeTraversal) null, (Scope.Var) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.getInitializingReferenceForConstants();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jcomp.ReferenceCollectingCallback$1", "com.google.javascript.jcomp.ReferenceCollectingCallback$1");
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) referenceCollectingCallback0);
      assertEquals(1, Node.DECR_FLAG);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = referenceCollectingCallback0.getReferenceCollection((Scope.Var) null);
      assertNull(referenceCollectingCallback_ReferenceCollection0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = Node.newNumber(1395.0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(120);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock0.provablyExecutesBefore(referenceCollectingCallback_BasicBlock1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(115, 115, 115);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(77);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(98, 120, (-2627));
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(101);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(108, (-773), (-773));
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node node0 = new Node((byte)113, (byte)113, (byte)113);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Node node0 = new Node(114, 114, 120);
      Compiler compiler0 = new Compiler();
      ControlFlowAnalysis controlFlowAnalysis0 = new ControlFlowAnalysis(compiler0, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, controlFlowAnalysis0);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node[] nodeArray0 = new Node[1];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      nodeArray0[0] = node0;
      Node node1 = new Node(100, nodeArray0, 114, 48);
      referenceCollectingCallback0.visit(nodeTraversal0, node0, node1);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(111, 72, 108);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isWellDefined();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isEscaped();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = referenceCollectingCallback_ReferenceCollection0.getInitializingReferenceForConstants();
      assertNull(referenceCollectingCallback_Reference0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null, (NodeTraversal) null, (Scope.Var) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isNeverAssigned();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null, (NodeTraversal) null, (Scope.Var) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isNeverAssigned();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null, (NodeTraversal) null, (Scope.Var) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(101);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node[] nodeArray0 = new Node[1];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      nodeArray0[0] = node0;
      Node node1 = new Node(102, nodeArray0, 1, 48);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(105, 105, 105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isVarDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, 6224, 118);
      LineNumberCheck lineNumberCheck0 = new LineNumberCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, lineNumberCheck0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isVarDeclaration();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(105, 105, 105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, 118, 118);
      LineNumberCheck lineNumberCheck0 = new LineNumberCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, lineNumberCheck0);
      node0.addChildToFront(node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = new Node(105, 105, 105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      Node node1 = referenceCollectingCallback_Reference0.getAssignedValue();
      assertEquals(40, Node.ORIGINALNAME_PROP);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node node0 = Node.newString(55, "com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection");
      Node node1 = new Node(101, node0, node0, node0, node0, 5, 3);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node1);
      boolean boolean0 = referenceCollectingCallback_Reference0.isLvalue();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0);
      Node node0 = compiler0.parseSyntheticCode((String) null, "U");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, referenceCollectingCallback_BasicBlock0, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isSimpleAssignmentToName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, 118, 118);
      LineNumberCheck lineNumberCheck0 = new LineNumberCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, lineNumberCheck0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.isLvalue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(118, 118, 118);
      LineNumberCheck lineNumberCheck0 = new LineNumberCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, lineNumberCheck0);
      node0.addChildToFront(node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isLvalue();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      Node[] nodeArray0 = new Node[1];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      nodeArray0[0] = node0;
      Node node1 = new Node(102, nodeArray0, 1, 48);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node1, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isLvalue();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = new Node(120);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock0.provablyExecutesBefore(referenceCollectingCallback_BasicBlock0);
      assertTrue(boolean0);
  }
}