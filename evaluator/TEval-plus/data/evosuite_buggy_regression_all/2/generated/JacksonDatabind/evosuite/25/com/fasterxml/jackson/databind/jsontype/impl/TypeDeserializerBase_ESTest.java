/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:58:11 GMT 2023
 */

package com.fasterxml.jackson.databind.jsontype.impl;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.BeanProperty;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.deser.AbstractDeserializer;
import com.fasterxml.jackson.databind.deser.BuilderBasedDeserializer;
import com.fasterxml.jackson.databind.jsontype.TypeIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.AsExternalTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.AsWrapperTypeDeserializer;
import com.fasterxml.jackson.databind.jsontype.impl.ClassNameIdResolver;
import com.fasterxml.jackson.databind.jsontype.impl.MinimalClassNameIdResolver;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.annotation.Annotation;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeDeserializerBase_ESTest extends TypeDeserializerBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<Map> class0 = Map.class;
      JavaType javaType0 = TypeFactory.unknownType();
      MapType mapType0 = typeFactory0.constructMapType(class0, javaType0, javaType0);
      ClassNameIdResolver classNameIdResolver0 = new ClassNameIdResolver(mapType0, typeFactory0);
      AsWrapperTypeDeserializer asWrapperTypeDeserializer0 = new AsWrapperTypeDeserializer(mapType0, classNameIdResolver0, "Od]", false, class0);
      String string0 = asWrapperTypeDeserializer0.getPropertyName();
      assertEquals("Od]", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<AbstractDeserializer> class0 = AbstractDeserializer.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, ",lKF9k;DxZn,z6Z2t8", true, class0);
      TypeIdResolver typeIdResolver0 = asPropertyTypeDeserializer0.getTypeIdResolver();
      assertNull(typeIdResolver0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Annotation> class0 = Annotation.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "zEIz#b,sqM^tl_ouYwP", false, class0);
      String string0 = asPropertyTypeDeserializer0.toString();
      assertEquals("[com.fasterxml.jackson.databind.jsontype.impl.AsPropertyTypeDeserializer; base-type:[simple type, class java.lang.Object]; id-resolver: null]", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Integer> class0 = Integer.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "W806Q(L&!_|#", true, class0);
      String string0 = asPropertyTypeDeserializer0.baseTypeName();
      assertEquals("java.lang.Object", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Integer> class0 = Integer.class;
      JsonTypeInfo.As jsonTypeInfo_As0 = JsonTypeInfo.As.EXISTING_PROPERTY;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "^YiO%", true, class0, jsonTypeInfo_As0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AsExternalTypeDeserializer asExternalTypeDeserializer0 = null;
      try {
        asExternalTypeDeserializer0 = new AsExternalTypeDeserializer((AsExternalTypeDeserializer) null, (BeanProperty) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Annotation> class0 = Annotation.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "zEIz#b,sqM^tl_ouYwP", false, class0);
      Class<?> class1 = asPropertyTypeDeserializer0.getDefaultImpl();
      assertNotNull(class1);
      assertEquals(1537, class1.getModifiers());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "Failed to parse @JsonSerializableSchema.schemaObjectPropertiesDefinition value", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, "Failed to parse @JsonSerializableSchema.schemaObjectPropertiesDefinition value");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<BuilderBasedDeserializer> class0 = BuilderBasedDeserializer.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "Failed to parse @JsonSerializableSchema.schemaObjectPropertiesDefinition value", true, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Object> class0 = Object.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "Failed to parse @JsonSerializableSchema.schemaObjectPropertiesDefinition value", true, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._deserializeWithNativeTypeId((JsonParser) null, (DeserializationContext) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      Class<Annotation> class0 = Annotation.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, (TypeIdResolver) null, "zEIztb,sqM^tl_ouYwP", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "zEIztb,sqM^tl_ouYwP", (TypeIdResolver) null, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JavaType javaType0 = TypeFactory.unknownType();
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      MinimalClassNameIdResolver minimalClassNameIdResolver0 = new MinimalClassNameIdResolver(javaType0, typeFactory0);
      Class<ClassNameIdResolver> class0 = ClassNameIdResolver.class;
      AsPropertyTypeDeserializer asPropertyTypeDeserializer0 = new AsPropertyTypeDeserializer(javaType0, minimalClassNameIdResolver0, "", false, class0);
      // Undeclared exception!
      try { 
        asPropertyTypeDeserializer0._handleUnknownTypeId((DeserializationContext) null, "", minimalClassNameIdResolver0, javaType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.jsontype.impl.TypeDeserializerBase", e);
      }
  }
}