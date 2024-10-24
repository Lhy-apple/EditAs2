/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:08:28 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      assertEquals(45, Node.IS_VAR_ARGS_PARAM);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      Node node1 = compiler0.parseSyntheticCode("function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      Node node2 = compiler0.parseSyntheticCode("function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      Node node3 = new Node(0, node1, node2, node0, node0);
      // Undeclared exception!
      try { 
        functionRewriter0.process(node3, node3);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      compiler0.parseSyntheticCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_value}}", "1H.-vm&.|}JzHYVD");
      functionRewriter0.process(node0, node0);
      assertEquals(7, Node.LOCAL_PROP);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_set(JSCompiler_set_name) {  return function(JSCompiler_set_value) {this[JSCompiler_set_name] = JSCompiler_set_value}}", "function JSCompiler_set(JSCompiler_set_name) {  return function(JSCompiler_set_value) {this[JSCompiler_set_name] = JSCompiler_set_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_returnArg(JSCompiler_returnArg_value) {  return function() {return JSCompiler_returnArg_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(35, Node.PARENTHESIZED_PROP);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_v>lue}}", "function JSCompiler_dentityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identityFn_v>lue}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertFalse(node0.isOnlyModifiesThisCall());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("function JSCompiler_identityFn() {  return function(JSCompiler_identityFn_value) {return JSCompiler_identtyFn_value}}");
      FunctionRewriter functionRewriter0 = new FunctionRewriter(compiler0);
      functionRewriter0.process(node0, node0);
      assertEquals(27, Node.SPECIALCALL_PROP);
  }
}
