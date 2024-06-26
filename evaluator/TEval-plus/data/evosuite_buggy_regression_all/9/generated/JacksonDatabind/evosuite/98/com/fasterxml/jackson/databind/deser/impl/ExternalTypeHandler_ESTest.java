/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:58:20 GMT 2023
 */

package com.fasterxml.jackson.databind.deser.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.deser.SettableBeanProperty;
import com.fasterxml.jackson.databind.deser.ValueInstantiator;
import com.fasterxml.jackson.databind.deser.impl.BeanPropertyMap;
import com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler;
import com.fasterxml.jackson.databind.deser.impl.ObjectIdReader;
import com.fasterxml.jackson.databind.deser.impl.PropertyBasedCreator;
import com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ExternalTypeHandler_ESTest extends ExternalTypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder((JavaType) null);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      boolean boolean0 = externalTypeHandler0.handleTypePropertyValue((JsonParser) null, (DeserializationContext) null, "com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler", externalTypeHandler_Builder0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = ExternalTypeHandler.builder((JavaType) null);
      // Undeclared exception!
      try { 
        externalTypeHandler_Builder0.addExternal((SettableBeanProperty) null, (TypeDeserializer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.ExternalTypeHandler$ExtTypedProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = ExternalTypeHandler.builder((JavaType) null);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      ExternalTypeHandler externalTypeHandler1 = externalTypeHandler0.start();
      assertNotSame(externalTypeHandler1, externalTypeHandler0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder((JavaType) null);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      boolean boolean0 = externalTypeHandler0.handlePropertyValue((JsonParser) null, (DeserializationContext) null, "y1$C(B:y(7+nnyjtb", externalTypeHandler_Builder0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder((JavaType) null);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      Object object0 = externalTypeHandler0.complete((JsonParser) null, (DeserializationContext) null, (Object) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ExternalTypeHandler.Builder externalTypeHandler_Builder0 = new ExternalTypeHandler.Builder((JavaType) null);
      ExternalTypeHandler externalTypeHandler0 = externalTypeHandler_Builder0.build((BeanPropertyMap) null);
      Class<Integer> class0 = Integer.class;
      ValueInstantiator.Base valueInstantiator_Base0 = new ValueInstantiator.Base(class0);
      SettableBeanProperty[] settableBeanPropertyArray0 = new SettableBeanProperty[0];
      PropertyBasedCreator propertyBasedCreator0 = PropertyBasedCreator.construct((DeserializationContext) null, (ValueInstantiator) valueInstantiator_Base0, settableBeanPropertyArray0, true);
      PropertyValueBuffer propertyValueBuffer0 = propertyBasedCreator0.startBuilding((JsonParser) null, (DeserializationContext) null, (ObjectIdReader) null);
      // Undeclared exception!
      try { 
        externalTypeHandler0.complete((JsonParser) null, (DeserializationContext) null, propertyValueBuffer0, propertyBasedCreator0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.deser.impl.PropertyValueBuffer", e);
      }
  }
}
