/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:02:30 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.PreprocessorSymbolTable;
import com.google.javascript.jscomp.ScopedAliases;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ScopedAliases_ESTest extends ScopedAliases_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, (CompilerOptions.AliasTransformationHandler) null);
      scopedAliases0.hotSwapScript(node0, node0);
      assertEquals(57, Node.REFLECTED_OBJECT);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, (PreprocessorSymbolTable) null, (CompilerOptions.AliasTransformationHandler) null);
      Node node0 = new Node(1971);
      scopedAliases0.process(node0, node0);
      assertFalse(node0.isScript());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(2147483645);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      Node node1 = new Node(37, node0, node0, node0, node0, 36, 2);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, (CompilerOptions.AliasTransformationHandler) null);
      scopedAliases0.hotSwapScript(node0, node1);
      assertEquals(15, Node.NO_SIDE_EFFECTS);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      Node node1 = new Node(30, node0, node0, node0);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node1);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, (CompilerOptions.AliasTransformationHandler) null);
      scopedAliases0.hotSwapScript(node1, node0);
      assertFalse(node0.isScript());
  }
}
