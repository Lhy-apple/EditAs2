/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:23:50 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.PropertyMetadata;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.deser.CreatorProperty;
import com.fasterxml.jackson.databind.introspect.AnnotatedParameter;
import com.fasterxml.jackson.databind.introspect.AnnotationMap;
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import java.time.chrono.ChronoLocalDate;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AsWrapperTypeDeserializer_ESTest extends AsWrapperTypeDeserializer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<TypeIdResolver> class0 = TypeIdResolver.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, ",", false, class0);
      PropertyName propertyName0 = PropertyName.NO_NAME;
      AnnotationMap annotationMap0 = new AnnotationMap();
      Integer integer0 = new Integer(1);
      CreatorProperty creatorProperty0 = new CreatorProperty(propertyName0, simpleType0, propertyName0, asWrapperTypeDeserializer0, annotationMap0, (AnnotatedParameter) null, 1, integer0, (PropertyMetadata) null);
      assertEquals((-1), creatorProperty0.getPropertyIndex());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("n4<tBc4Q`Ny'R2 g}|");
      Class<Object> class0 = Object.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "n4<tBc4Q`Ny'R2 g}|", false, class0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromScalar(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<ArrayNode> class0 = ArrayNode.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "~{E6", true, class0);
      JsonTypeInfo.As jsonTypeInfo_As0 = asWrapperTypeDeserializer0.getTypeInclusion();
      assertEquals(JsonTypeInfo.As.WRAPPER_OBJECT, jsonTypeInfo_As0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("[Qdj2");
      Class<ChronoLocalDate> class0 = ChronoLocalDate.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "[Qdj2", false, class0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromArray(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("JSON");
      Class<ObjectWriter> class0 = ObjectWriter.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "JSON", true, class0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromAny(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      JsonFactory jsonFactory0 = new JsonFactory();
      JsonParser jsonParser0 = jsonFactory0.createParser("Incompatible types: declared root type (");
      Class<ObjectWriter> class0 = ObjectWriter.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "Incompatible types: declared root type (", true, class0);
      // Undeclared exception!
      try { 
        asWrapperTypeDeserializer0.deserializeTypedFromObject(jsonParser0, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SimpleType simpleType0 = (SimpleType)TypeBindings.UNBOUND;
      Class<ArrayNode> class0 = ArrayNode.class;
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(simpleType0, (TypeIdResolver) null, "~{E6", true, class0);
      TypeDeserializer typeDeserializer0 = asWrapperTypeDeserializer0.forProperty((BeanProperty) null);
      assertSame(typeDeserializer0, asWrapperTypeDeserializer0);
  }
}