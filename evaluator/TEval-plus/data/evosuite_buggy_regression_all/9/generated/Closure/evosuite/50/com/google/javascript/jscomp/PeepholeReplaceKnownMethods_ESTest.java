/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:50:54 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckRegExp;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FlowSensitiveInlineVariables;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.PeepholeReplaceKnownMethods;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PeepholeReplaceKnownMethods_ESTest extends PeepholeReplaceKnownMethods_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = Node.newString("1");
      Node node1 = peepholeReplaceKnownMethods0.optimizeSubtree(node0);
      assertEquals(25, Node.ISNUMBER_PROP);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = new Node(37, 37, 37);
      Node node1 = peepholeReplaceKnownMethods0.optimizeSubtree(node0);
      assertEquals(51, Node.STATIC_SOURCE_FILE);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Compiler compiler0 = new Compiler();
      FlowSensitiveInlineVariables flowSensitiveInlineVariables0 = new FlowSensitiveInlineVariables(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, flowSensitiveInlineVariables0);
      peepholeReplaceKnownMethods0.beginTraversal(nodeTraversal0);
      Node node0 = new Node(4093, 4093, 4093);
      Node node1 = new Node(37, node0, 2, 43);
      Node node2 = peepholeReplaceKnownMethods0.optimizeSubtree(node1);
      assertFalse(node2.isOptionalArg());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PeepholeReplaceKnownMethods peepholeReplaceKnownMethods0 = new PeepholeReplaceKnownMethods();
      Node node0 = new Node(4180, 4180, 4180);
      Compiler compiler0 = new Compiler();
      CheckRegExp checkRegExp0 = new CheckRegExp(compiler0);
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkRegExp0, (ScopeCreator) null);
      peepholeReplaceKnownMethods0.beginTraversal(nodeTraversal0);
      Node node1 = new Node(37, node0, 23, 43);
      Node node2 = peepholeReplaceKnownMethods0.optimizeSubtree(node1);
      assertEquals(43, node2.getCharno());
  }
}