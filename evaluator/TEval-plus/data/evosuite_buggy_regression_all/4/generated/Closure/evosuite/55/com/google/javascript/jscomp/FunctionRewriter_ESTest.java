/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:14:40 GMT 2023
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
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(0, Node.NON_SPECIALCALL);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_emptyFn() {  return function() {}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(47, Node.IS_DISPATCHER);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_set(JSCompiler_set_name) {  return function(JSCompiler_set_value) {this[JSCompiler_set_name] = JSCompiler_set_value}}", "function JSCompiler_set(JSCompiler_set_name) {  return function(JSCompiler_set_value) {this[JSCompiler_set_name] = JSCompiler_set_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertFalse(node0.isFromExterns());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_get(JSCompiler_get_name) {  return function() {return this[JSCompiler_get_name]}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(22, Node.TARGETBLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_v4lue) {return JSCompiler_identityFn_va+ue}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(1, Node.FLAG_GLOBAL_STATE_UNMODIFIED);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityF_value}}", "function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityF_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(53, Node.LAST_PROP);
  }
}