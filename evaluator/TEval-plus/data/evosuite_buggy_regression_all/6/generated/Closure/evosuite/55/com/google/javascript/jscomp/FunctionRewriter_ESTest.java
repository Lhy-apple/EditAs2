/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:56:31 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.FunctionRewriter;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionRewriter_ESTest extends FunctionRewriter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_emptyFn() {  return function() {}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(44, Node.IS_OPTIONAL_PARAM);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      Node node1 = compiler0.parseSyntheticCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "aUH!#LG<:Iiyz");
      functionRewriter0.process(node1, node0);
      assertFalse(node0.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_set(JSCompiler_set_name) {  return function(JSCompiler_set_value) {this[JSCompiler_set_name] = JSCompiler_set_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(2, Node.RIGHT);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      Node node0 = compiler0.parseTestCode("function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSComp$ler_returnArg_value}}");
      functionRewriter0.process(node0, node0);
      assertFalse(node0.isQuotedString());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_v.lue}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_idntityFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(2, Node.RIGHT);
  }
}
