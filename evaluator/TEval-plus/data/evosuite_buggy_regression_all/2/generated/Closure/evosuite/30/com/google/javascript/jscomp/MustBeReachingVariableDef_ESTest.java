/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:27:37 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.MustBeReachingVariableDef;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import java.util.ArrayDeque;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MustBeReachingVariableDef_ESTest extends MustBeReachingVariableDef_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Node node0 = new Node(167);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(167);
      Node node1 = mustBeReachingVariableDef0.getDef("Not declared as a type name", node0);
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Stack<Scope.Var> stack0 = new Stack<Scope.Var>();
      ListIterator<Scope.Var> listIterator0 = stack0.listIterator();
      stack0.add((Scope.Var) null);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = null;
      try {
        mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef(listIterator0);
        fail("Expecting exception: ConcurrentModificationException");
      
      } catch(ConcurrentModificationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Vector$Itr", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ArrayDeque<Scope.Var> arrayDeque0 = new ArrayDeque<Scope.Var>();
      Iterator<Scope.Var> iterator0 = arrayDeque0.iterator();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef(iterator0);
      boolean boolean0 = mustBeReachingVariableDef_MustDef0.equals(arrayDeque0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Node node0 = new Node((-901), (-901), (-901));
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = mustBeReachingVariableDef0.createEntryLattice();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.join(mustBeReachingVariableDef_MustDef0, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef0, mustBeReachingVariableDef_MustDef1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = new Node(98);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(39);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = new Node(99, 99, 99);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(2);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Node node0 = new Node(100, 159, 159);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(15);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = new Node(102, 102, 102);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(16);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Node node0 = new Node(108, 108, 108);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(31);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Node node0 = new Node(109);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(38);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = new Node(111);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(50);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = new Node(112);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(52);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(113, 113, 113);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(1389);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = new Node(115);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Compiler compiler0 = new Compiler();
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Node node0 = new Node(90, 90, 90);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.analyze(2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Node node0 = new Node(115, 115, 115);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Compiler compiler0 = new Compiler();
      JSType[] jSTypeArray0 = new JSType[1];
      jSTypeArray0[0] = (JSType) objectType0;
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node1, true, true);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(54);
  }
}
