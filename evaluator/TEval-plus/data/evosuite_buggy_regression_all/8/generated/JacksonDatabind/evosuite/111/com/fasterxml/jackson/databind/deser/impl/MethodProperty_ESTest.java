/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:21:52 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.impl.MethodProperty;
import java.lang.reflect.Method;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MethodProperty_ESTest extends MethodProperty_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MethodProperty methodProperty0 = null;
      try {
        methodProperty0 = new MethodProperty((MethodProperty) null, (Method) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      MethodProperty methodProperty0 = null;
      try {
        methodProperty0 = new MethodProperty((MethodProperty) null, propertyName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.ConcreteBeanPropertyBase", e);
      }
  }
}