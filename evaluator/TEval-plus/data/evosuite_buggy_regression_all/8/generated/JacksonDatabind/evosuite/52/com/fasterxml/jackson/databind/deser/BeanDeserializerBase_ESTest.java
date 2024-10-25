/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:10:27 GMT 2023
 */

package com.fasterxml.jackson.databind.deser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.cfg.ContextAttributes;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import java.lang.reflect.InvocationTargetException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BeanDeserializerBase_ESTest extends BeanDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.FIELD;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.ANY;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      ContextAttributes contextAttributes0 = ContextAttributes.Impl.getEmpty();
      Class<AsWrapperTypeDeserializer> class0 = AsWrapperTypeDeserializer.class;
      try { 
        objectMapper1.convertValue((Object) contextAttributes0, class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not construct instance of com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer: no suitable constructor found, can not deserialize from Object value (missing default constructor or creator, or perhaps need to add/enable type information?)
         //  at [Source: java.lang.String@0000000829; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.databind.ObjectMapper", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.FIELD;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.ANY;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      objectMapper1.enableDefaultTyping();
      ObjectReader objectReader0 = objectMapper1.reader();
      Class<InvocationTargetException> class0 = InvocationTargetException.class;
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertFalse(objectReader1.equals((Object)objectReader0));
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ObjectMapper objectMapper0 = new ObjectMapper();
      PropertyAccessor propertyAccessor0 = PropertyAccessor.FIELD;
      JsonAutoDetect.Visibility jsonAutoDetect_Visibility0 = JsonAutoDetect.Visibility.ANY;
      ObjectMapper objectMapper1 = objectMapper0.setVisibility(propertyAccessor0, jsonAutoDetect_Visibility0);
      ObjectReader objectReader0 = objectMapper1.reader();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      ObjectReader objectReader1 = objectReader0.forType(class0);
      assertFalse(objectReader1.equals((Object)objectReader0));
  }
}
