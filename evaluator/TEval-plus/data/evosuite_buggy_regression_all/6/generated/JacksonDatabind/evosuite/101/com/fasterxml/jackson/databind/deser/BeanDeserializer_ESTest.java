/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:30:04 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerBase;
import java.util.Locale;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializer_ESTest extends BeanDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Locale locale0 = Locale.ITALIAN;
      Set<String> set0 = locale0.getUnicodeLocaleKeys();
      BeanDeserializer beanDeserializer0 = null;
      try {
        beanDeserializer0 = new BeanDeserializer((BeanDeserializerBase) null, set0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.BeanDeserializerBase", e);
      }
  }
}