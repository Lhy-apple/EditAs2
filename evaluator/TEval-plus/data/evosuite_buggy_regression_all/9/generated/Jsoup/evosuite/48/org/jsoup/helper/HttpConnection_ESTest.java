/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:09:27 GMT 2023
 */

package org.jsoup.helper;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.UnknownServiceException;
import java.nio.charset.IllegalCharsetNameException;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.testdata.EvoSuiteURL;
import org.evosuite.runtime.testdata.NetworkHandling;
import org.jsoup.Connection;
import org.jsoup.helper.HttpConnection;
import org.jsoup.parser.Parser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HttpConnection_ESTest extends HttpConnection_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      Connection connection1 = connection0.followRedirects(false);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.ignoreHttpErrors(true);
      try { 
        connection1.get();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Could not find: 
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F");
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http:_F");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "Content-Encoding");
      Connection.Response connection_Response0 = connection0.execute();
      connection_Response0.parse();
      String string0 = connection_Response0.body();
      assertEquals("Content-Encoding", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_m/");
      Connection.Request connection_Request0 = connection0.request();
      boolean boolean0 = connection_Request0.ignoreContentType();
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertTrue(connection_Request0.validateTLSCertificates());
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertFalse(boolean0);
      assertEquals(3000, connection_Request0.timeout());
      assertTrue(connection_Request0.followRedirects());
      assertEquals(1048576, connection_Request0.maxBodySize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      Parser parser0 = Parser.htmlParser();
      Connection connection1 = connection0.parser(parser0);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.ignoreContentType(false);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F");
      Connection.Request connection_Request0 = connection0.request();
      boolean boolean0 = connection_Request0.validateTLSCertificates();
      assertTrue(connection_Request0.followRedirects());
      assertFalse(connection_Request0.ignoreContentType());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertEquals(3000, connection_Request0.timeout());
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      Connection connection1 = connection0.validateTLSCertificates(true);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_f");
      Connection.Request connection_Request0 = connection0.request();
      boolean boolean0 = connection_Request0.followRedirects();
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertFalse(connection_Request0.ignoreContentType());
      assertTrue(connection_Request0.validateTLSCertificates());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertTrue(boolean0);
      assertEquals(3000, connection_Request0.timeout());
      assertFalse(connection_Request0.ignoreHttpErrors());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.removeCookie((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Cookie name must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      Connection connection1 = connection0.response((Connection.Response) httpConnection_Response0);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.maxBodySize(389);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      URL uRL0 = MockURL.getFileExample();
      connection0.url(uRL0);
      try { 
        connection0.get();
        fail("Expecting exception: MalformedURLException");
      
      } catch(MalformedURLException e) {
         //
         // Only http & https protocols supported
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.userAgent("http:_F/");
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F");
      Connection connection1 = connection0.postDataCharset("US-ASCII");
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_/");
      Connection.Response connection_Response0 = connection0.response();
      assertNull(connection_Response0.charset());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_/");
      Connection connection1 = connection0.timeout(200);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_/");
      Connection connection1 = connection0.data("http:_/", "http:_/");
      connection1.data("http:_/", "Content-Encoding");
      try { 
        connection0.get();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_5F/");
      Connection.Method connection_Method0 = Connection.Method.PATCH;
      Connection connection1 = connection0.method(connection_Method0);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.request((Connection.Request) null);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.header("http:_F/", "Content-Encoding");
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.referrer("http:_F/");
      Connection connection2 = connection1.referrer("Content-Encoding");
      assertSame(connection2, connection0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      URL uRL0 = MockURL.getHttpExample();
      Connection connection0 = HttpConnection.connect(uRL0);
      assertNotNull(connection0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.statusMessage();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      // Undeclared exception!
      try { 
        httpConnection_Response0.bodyAsBytes();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Request must be executed (with .execute(), .get(), or .post() before getting response body
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      String string0 = httpConnection_Response0.charset();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      int int0 = httpConnection_Response0.statusCode();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("LbR%0", "LbR%0");
      InputStream inputStream0 = httpConnection_KeyVal0.inputStream();
      assertNull(inputStream0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("qF<=LB", "qF<=LB");
      String string0 = httpConnection_KeyVal0.toString();
      assertEquals("qF<=LB=qF<=LB", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("Content-Encoding", "Content-Encoding");
      Connection connection1 = connection0.data((Map<String, String>) hashMap0);
      assertSame(connection1, connection0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      String[] stringArray0 = new String[1];
      // Undeclared exception!
      try { 
        connection0.data(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must supply an even number of key value pairs
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      String[] stringArray0 = new String[0];
      Connection connection1 = connection0.data(stringArray0);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      String[] stringArray0 = new String[4];
      // Undeclared exception!
      try { 
        connection0.data(stringArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Data key must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      LinkedHashSet<Connection.KeyVal> linkedHashSet0 = new LinkedHashSet<Connection.KeyVal>();
      HttpConnection.KeyVal httpConnection_KeyVal0 = HttpConnection.KeyVal.create("pre", "t}?n1{ia3}1cO");
      linkedHashSet0.add(httpConnection_KeyVal0);
      Connection connection1 = connection0.data((Collection<Connection.KeyVal>) linkedHashSet0);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.putIfAbsent("Content-Encoding", "http:_F/");
      Connection connection1 = connection0.cookies(hashMap0);
      assertSame(connection0, connection1);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F");
      Connection.Request connection_Request0 = connection0.request();
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      HttpConnection.Response httpConnection_Response1 = (HttpConnection.Response)httpConnection_Response0.cookie("Content-Encoding", "Content-Encoding");
      try { 
        HttpConnection.Response.execute(connection_Request0, httpConnection_Response1);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_5F/");
      // Undeclared exception!
      try { 
        connection0.timeout((-1078));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Timeout milliseconds must be 0 (infinite) or greater
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      // Undeclared exception!
      try { 
        connection0.maxBodySize((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // maxSize must be 0 (unlimited) or larger
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      // Undeclared exception!
      try { 
        connection0.postDataCharset("Content-Encoding");
        fail("Expecting exception: IllegalCharsetNameException");
      
      } catch(IllegalCharsetNameException e) {
         //
         // Content-Encoding
         //
         verifyException("org.jsoup.helper.HttpConnection$Request", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("https:_F/");
      Connection.Request connection_Request0 = connection0.request();
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      try { 
        HttpConnection.Response.execute(connection_Request0, httpConnection_Response0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F");
      EvoSuiteURL evoSuiteURL0 = new EvoSuiteURL("http:_F");
      NetworkHandling.createRemoteTextFile(evoSuiteURL0, "http:_F");
      Connection.Request connection_Request0 = connection0.request();
      HttpConnection.Response httpConnection_Response0 = HttpConnection.Response.execute(connection_Request0);
      String string0 = httpConnection_Response0.body();
      assertEquals(3000, connection_Request0.timeout());
      assertFalse(connection_Request0.ignoreHttpErrors());
      assertTrue(connection_Request0.validateTLSCertificates());
      assertTrue(connection_Request0.followRedirects());
      assertEquals("UTF-8", connection_Request0.postDataCharset());
      assertEquals(1048576, connection_Request0.maxBodySize());
      assertEquals("http:_F", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put((String) null, linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.statusMessage());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("Zw81kRwyu?#.Sf5}8", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.contentType());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("set-cookie");
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.charset());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add((String) null);
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertFalse(hashMap0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("");
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("set-cookie", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertEquals(1, hashMap0.size());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("Zw81kRwyu?#.Sf5}8");
      HashMap<String, List<String>> hashMap0 = new HashMap<String, List<String>>();
      hashMap0.put("Zw81kRwyu?#.Sf5}8", linkedList0);
      HttpConnection.Response httpConnection_Response0 = new HttpConnection.Response();
      httpConnection_Response0.processResponseHeaders(hashMap0);
      assertNull(httpConnection_Response0.statusMessage());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:_F/");
      Connection connection1 = connection0.data("http:_F/", "http:_F/");
      try { 
        connection1.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      connection0.data("http:/", "figcaption", (InputStream) sequenceInputStream0);
      try { 
        connection0.post();
        fail("Expecting exception: UnknownServiceException");
      
      } catch(UnknownServiceException e) {
         //
         // protocol doesn't support output
         //
         verifyException("org.evosuite.runtime.mock.java.net.EvoHttpURLConnection", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Connection connection0 = HttpConnection.connect("http:/");
      Connection connection1 = connection0.cookie("Content-Encoding", "Content-Encoding");
      connection1.cookie("http:/", "Content-Encoding");
      try { 
        connection1.get();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // HTTP error fetching URL
         //
         verifyException("org.jsoup.helper.HttpConnection$Response", e);
      }
  }
}