/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:46:34 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NameAnonymousFunctionsMapped;
import com.google.javascript.jscomp.RenamePrototypes;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RenamePrototypes_ESTest extends RenamePrototypes_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, true, (char[]) null, (VariableMap) null);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefaultPassConfig$80");
      renamePrototypes0.process(node0, node0);
      assertEquals(42, Node.IS_CONSTANT_NAME);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      VariableMap variableMap0 = compiler0.getPropertyMap();
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, false, (char[]) null, variableMap0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefaultPassConfig$80");
      renamePrototypes0.process(node0, node0);
      assertEquals((-1), Node.CATCH_SCOPE_PROP);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      NameAnonymousFunctionsMapped nameAnonymousFunctionsMapped0 = new NameAnonymousFunctionsMapped(compiler0);
      VariableMap variableMap0 = nameAnonymousFunctionsMapped0.getFunctionMap();
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, false, (char[]) null, variableMap0);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefaultPassConfig$80");
      renamePrototypes0.process(node0, node0);
      assertEquals(1, Node.FLAG_GLOBAL_STATE_UNMODIFIED);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, false, (char[]) null, (VariableMap) null);
      Node[] nodeArray0 = new Node[0];
      Node node0 = new Node(35, nodeArray0);
      // Undeclared exception!
      try { 
        renamePrototypes0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AbstractCompiler.LifeCycleStage abstractCompiler_LifeCycleStage0 = AbstractCompiler.LifeCycleStage.NORMALIZED;
      compiler0.setLifeCycleStage(abstractCompiler_LifeCycleStage0);
      Node node0 = new Node(22);
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, false, (char[]) null, (VariableMap) null);
      Node node1 = new Node(64, node0);
      // Undeclared exception!
      try { 
        renamePrototypes0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      RenamePrototypes renamePrototypes0 = new RenamePrototypes(compiler0, true, (char[]) null, (VariableMap) null);
      VariableMap variableMap0 = renamePrototypes0.getPropertyMap();
      assertNotNull(variableMap0);
  }
}
