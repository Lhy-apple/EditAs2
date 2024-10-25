/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:10:09 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.MustBeReachingVariableDef;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.rhino.Node;
import java.util.HashSet;
import java.util.Iterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MustBeReachingVariableDef_ESTest extends MustBeReachingVariableDef_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "Dsr0odV$Mh", "Dsr0odV$Mh");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      mustBeReachingVariableDef0.analyze(15);
      Node node1 = mustBeReachingVariableDef0.getDef("Dsr0odV$Mh", node0);
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HashSet<Scope.Var> hashSet0 = new HashSet<Scope.Var>();
      hashSet0.add((Scope.Var) null);
      Iterator<Scope.Var> iterator0 = hashSet0.iterator();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = null;
      try {
        mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef(iterator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef$MustDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      boolean boolean0 = mustBeReachingVariableDef_MustDef0.equals("com.google.javascript.jscomp.MustBeReachingVariableDef$MustDefJoin");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com", "com");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.join(mustBeReachingVariableDef_MustDef0, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef0, mustBeReachingVariableDef_MustDef1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      Node node1 = Node.newString(98, "Expected setCompiler to be called first");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
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
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      Node node1 = Node.newString(104, "arguments");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "Dsr0odV$Mh", "Dsr0odV$Mh");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      Node node1 = Node.newString(105, "TWD_\"VEjWhF3bwI,j`");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "}RD1KFK{ex$", "}RD1KFK{ex$");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(106, "}RD1KFK{ex$");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "CUbehyd9JT", "/m");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, true);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(108, "O");
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
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
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "}RD1KFK{ex$", "L~Byqo3E,");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(109, "");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      Node node1 = Node.newString(111, "arguments");
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "}RD1KFK{ex$", "}RD1KFK{ex$");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(113, "");
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "%krS", "O");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(115, "%krS");
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // malformed 'for' statement FOR %krS
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "}RD1KFK{ex$", "}RD1KFK{ex$");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      Node node1 = Node.newString(116, "");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "J", "J");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Node node1 = new Node(118);
      Scope scope0 = new Scope(node1, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = mustBeReachingVariableDef0.createInitialEstimateLattice();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "J", "J");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Node node1 = new Node(120);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = mustBeReachingVariableDef0.createInitialEstimateLattice();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(121, ";If+_^:T3u. \"");
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Scope scope0 = new Scope(node0, compiler0);
      Node node1 = Node.newString(122, "arguments");
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = mustBeReachingVariableDef0.createInitialEstimateLattice();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "J", "J");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      Node node1 = new Node(125);
      Scope scope0 = new Scope(node1, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "%krS", "O");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = new MustBeReachingVariableDef.MustDef();
      Node node1 = Node.newString(87, "%krS");
      // Undeclared exception!
      try { 
        mustBeReachingVariableDef0.flowThrough(node1, mustBeReachingVariableDef_MustDef0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MustBeReachingVariableDef", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "arguments", "arguments");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, false, false);
      Scope scope0 = new Scope(node0, compiler0);
      MustBeReachingVariableDef mustBeReachingVariableDef0 = new MustBeReachingVariableDef(controlFlowGraph0, scope0, compiler0);
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef0 = mustBeReachingVariableDef0.createInitialEstimateLattice();
      MustBeReachingVariableDef.MustDef mustBeReachingVariableDef_MustDef1 = mustBeReachingVariableDef0.flowThrough(node0, mustBeReachingVariableDef_MustDef0);
      assertNotSame(mustBeReachingVariableDef_MustDef1, mustBeReachingVariableDef_MustDef0);
  }
}
