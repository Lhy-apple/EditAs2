/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:43:21 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler;
import com.fasterxml.jackson.databind.jsontype.impl.AsArrayTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ExternalTypeHandler_ESTest extends ExternalTypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver((JavaType) null, typeFactory0);
      AsArrayTypeDeserializer asArrayTypeDeserializer0 = new AsArrayTypeDeserializer((JavaType) null, classNameIdResolver0, "USE_LONG_FOR_INTS", false, (JavaType) null);
      // Undeclared exception!
      try { 
        externalTypeHandler_Builder0.addExternal((SettableBeanProperty) null, asArrayTypeDeserializer0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler$Builder", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ExternalTypeHandler externalTypeHandler0 = null;
      try {
        externalTypeHandler0 = new ExternalTypeHandler((ExternalTypeHandler) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", e);
      }
  }
}
