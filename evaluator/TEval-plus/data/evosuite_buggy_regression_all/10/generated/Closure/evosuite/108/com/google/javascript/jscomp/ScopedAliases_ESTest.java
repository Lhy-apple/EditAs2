/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:49:31 GMT 2023
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
      assertFalse(node0.isDelProp());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Compiler compiler0 = new Compiler();
      CompilerOptions.AliasTransformationHandler compilerOptions_AliasTransformationHandler0 = compilerOptions0.getAliasTransformationHandler();
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, (PreprocessorSymbolTable) null, compilerOptions_AliasTransformationHandler0);
      Node node0 = Node.newString("");
      scopedAliases0.process(node0, node0);
      assertNull(node0.getSourceFileName());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(12);
      Node node1 = new Node(37, node0, node0, node0, node0, 12, 4095);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node1);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, (CompilerOptions.AliasTransformationHandler) null);
      scopedAliases0.hotSwapScript(node1, node1);
      assertFalse(node1.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node((-644));
      Node node1 = new Node(42, node0, node0, node0, node0, 46, 48);
      CompilerOptions.AliasTransformationHandler compilerOptions_AliasTransformationHandler0 = CompilerOptions.NULL_ALIAS_TRANSFORMATION_HANDLER;
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, compilerOptions_AliasTransformationHandler0);
      scopedAliases0.hotSwapScript(node0, node0);
      assertFalse(node0.isLocalResultCall());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      Node node1 = new Node(46, node0, node0, node0, node0, 43, 29);
      PreprocessorSymbolTable preprocessorSymbolTable0 = new PreprocessorSymbolTable(node0);
      ScopedAliases scopedAliases0 = new ScopedAliases(compiler0, preprocessorSymbolTable0, (CompilerOptions.AliasTransformationHandler) null);
      scopedAliases0.hotSwapScript(node1, node1);
      assertFalse(node1.isArrayLit());
  }
}